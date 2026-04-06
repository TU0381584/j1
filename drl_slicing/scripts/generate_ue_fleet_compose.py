#!/usr/bin/env python3
"""
Generate a UERANSIM UE fleet docker-compose file and profile CSV.

The generated compose file is intended to be used with docker_open5gs and creates
one NR-UE container per UE with unique IMSI/IMEI identities.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def _default_paths():
    root_dir = Path(__file__).resolve().parents[2]
    out_dir = root_dir / "docker_open5gs" / "generated"
    return out_dir / "nr-ue-fleet.yaml", out_dir / "ue_fleet_profiles.csv"


def _build_numeric_id(base_value: str, offset: int) -> str:
    width = len(base_value)
    value = int(base_value) + offset
    return f"{value:0{width}d}"


def _assign_profiles(ue_count: int, embb_ratio: float, seed: int) -> List[str]:
    rng = random.Random(seed)

    if ue_count <= 1:
        return ["embb"] * ue_count

    embb_count = int(round(ue_count * embb_ratio))
    embb_count = max(1, min(ue_count - 1, embb_count))
    urllc_count = ue_count - embb_count

    profiles = ["embb"] * embb_count + ["urllc"] * urllc_count
    rng.shuffle(profiles)
    return profiles


def _profile_to_slice(profile: str, ue_index: int) -> Tuple[str, str, str]:
    """
    Map UE profile to a deterministic S-NSSAI assignment.

    - URLLC UEs -> SD 000001 (slice 1-1)
    - eMBB UEs  -> split between SD 000000 (slice 1-0) and SD 000002 (slice 1-2)
    """
    sst = "1"
    if profile == "urllc":
        return sst, "000001", "1-1"
    if ue_index % 2 == 0:
        return sst, "000000", "1-0"
    return sst, "000002", "1-2"


def _write_compose(
    compose_path: Path,
    ue_count: int,
    profiles: List[str],
    imsi_base: str,
    imei_base: str,
    imeisv_base: str,
    ki: str,
    op: str,
    amf: str,
    gnb_ip_override: str,
) -> List[dict]:
    records = []

    lines: List[str] = []
    lines.append("version: '3'")
    lines.append("services:")

    for i in range(1, ue_count + 1):
        idx = f"{i:03d}"
        service_name = f"nr_ue_{idx}"
        component_name = f"ueransim-ue-{idx}"
        imsi = _build_numeric_id(imsi_base, i)
        imei = _build_numeric_id(imei_base, i)
        imeisv = _build_numeric_id(imeisv_base, i)
        profile = profiles[i - 1]
        sst, sd, slice_id = _profile_to_slice(profile, i)

        records.append(
            {
                "service_name": service_name,
                "container_name": service_name,
                "component_name": component_name,
                "imsi": imsi,
                "imei": imei,
                "imeisv": imeisv,
                "ki": ki,
                "op": op,
                "amf": amf,
                "profile": profile,
                "sst": sst,
                "sd": sd,
                "slice_id": slice_id,
            }
        )

        env_lines = [
            f"      - COMPONENT_NAME={component_name}",
            f"      - UE1_IMSI={imsi}",
            f"      - UE1_IMEI={imei}",
            f"      - UE1_IMEISV={imeisv}",
            f"      - UE1_KI={ki}",
            f"      - UE1_OP={op}",
            f"      - UE1_AMF={amf}",
            f"      - UE_PROFILE={profile}",
            f"      - UE_SLICE_SST={sst}",
            f"      - UE_SLICE_SD={sd}",
        ]
        if gnb_ip_override:
            env_lines.append(f"      - NR_GNB_IP={gnb_ip_override}")

        lines.extend(
            [
                f"  {service_name}:",
                "    image: docker_ueransim",
                f"    container_name: {service_name}",
                "    stdin_open: true",
                "    tty: true",
                "    volumes:",
                "      - ../ueransim:/mnt/ueransim",
                "      - /etc/timezone:/etc/timezone:ro",
                "      - /etc/localtime:/etc/localtime:ro",
                "    env_file:",
                "      - ../.env",
                "    environment:",
                *env_lines,
                "    expose:",
                '      - "4997/udp"',
                "    cap_add:",
                "      - NET_ADMIN",
                "    privileged: true",
                "    networks:",
                "      default: {}",
            ]
        )

    lines.extend(
        [
            "networks:",
            "  default:",
            "    external:",
            "      name: docker_open5gs_default",
        ]
    )

    compose_path.parent.mkdir(parents=True, exist_ok=True)
    compose_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def _write_profiles_csv(csv_path: Path, records: List[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "service_name",
                "container_name",
                "component_name",
                "imsi",
                "imei",
                "imeisv",
                "ki",
                "op",
                "amf",
                "profile",
                "sst",
                "sd",
                "slice_id",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    default_compose, default_profiles = _default_paths()

    parser = argparse.ArgumentParser(description="Generate 5G UE fleet compose + profile mapping")
    parser.add_argument("--ue-count", type=int, default=100, help="Number of UE containers")
    parser.add_argument("--embb-ratio", type=float, default=0.7, help="Fraction of eMBB UEs")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for profile assignment")

    parser.add_argument("--imsi-base", default="001010000100000", help="Base IMSI as integer string")
    parser.add_argument("--imei-base", default="356938035640000", help="Base IMEI as integer string")
    parser.add_argument("--imeisv-base", default="4370816125800000", help="Base IMEISV as integer string")

    parser.add_argument("--ki", default="465B5CE8B199B49FAA5F0A2EE238A6BC", help="UE Ki")
    parser.add_argument("--op", default="E8ED289DEBA952E4283B54E88E6183CA", help="UE OP/OPC")
    parser.add_argument("--amf", default="8000", help="UE AMF")
    parser.add_argument(
        "--gnb-ip-override",
        default="",
        help="Optional gNB IP to inject into each UE container (overrides env_file NR_GNB_IP)",
    )

    parser.add_argument("--output-compose", default=str(default_compose), help="Output docker-compose path")
    parser.add_argument("--output-profiles", default=str(default_profiles), help="Output profiles CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.ue_count <= 0:
        raise ValueError("--ue-count must be > 0")
    if not (0.0 <= args.embb_ratio <= 1.0):
        raise ValueError("--embb-ratio must be between 0 and 1")

    compose_path = Path(args.output_compose).resolve()
    csv_path = Path(args.output_profiles).resolve()

    profiles = _assign_profiles(args.ue_count, args.embb_ratio, args.random_seed)
    records = _write_compose(
        compose_path=compose_path,
        ue_count=args.ue_count,
        profiles=profiles,
        imsi_base=args.imsi_base,
        imei_base=args.imei_base,
        imeisv_base=args.imeisv_base,
        ki=args.ki,
        op=args.op,
        amf=args.amf,
        gnb_ip_override=args.gnb_ip_override,
    )
    _write_profiles_csv(csv_path, records)

    embb_count = sum(1 for record in records if record["profile"] == "embb")
    urllc_count = len(records) - embb_count

    print(f"Generated compose file: {compose_path}")
    print(f"Generated profile CSV : {csv_path}")
    print(f"UE count={len(records)} | eMBB={embb_count} | URLLC={urllc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
