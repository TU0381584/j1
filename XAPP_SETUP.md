# ORANSlice xApp Integration Setup

## Cloned Repositories

We now have all components needed for RAN slicing control via E2 interface:

### 1. **xApp-OAI** (`/home/w5/ors/xapp-oai`)
- **xApp BS Connector** (`xapp_bs_connector/`) — C++ bridge between xApp and RIC
- **base-xapp** (`base-xapp/`) — Python slicing control logic
- **Purpose**: Receives slicing policies from RIC and sends control commands to gNB

### 2. **E2Sim** (`/home/w5/ors/o-ran-e2sim`)
- Service Model (SM) simulator that encapsulates custom SM buffers
- Runs on the same server as gNB
- Communicates with gNB via UDP sockets
- **Purpose**: Bridges gNB E2 Agent and RIC E2 interface

## Deployment Order

For a complete RAN slicing test without OSC RIC:

### Option A: Direct gNB-to-xApp Communication (Simpler)
1. Start gNB with E2 Agent enabled
2. Start E2Sim (runs locally)
3. Start xApp BS Connector
4. Start xApp Logic Unit (Python)
5. Modify slicing policies in Python and see gNB respond

### Option B: With OSC Near-RT RIC (Full O-RAN)
Requires OpenShift cluster deployment — check OSC docs:
https://docs.o-ran-sc.org/projects/o-ran-sc-ric-plt-ric-dep/

## Next Steps

1. **Build E2Sim**:
   ```bash
   cd /home/w5/ors/o-ran-e2sim
   ./build_e2sim.sh
   ```

2. **Build xApp BS Connector**:
   ```bash
   cd /home/w5/ors/xapp-oai/xapp_bs_connector
   make
   ```

3. **Run E2Sim**:
   ```bash
   cd /home/w5/ors/o-ran-e2sim
   ./run_e2sim.sh <e2term-ip> <e2term-port>
   ```

4. **Run xApp Connector**:
   ```bash
   cd /home/w5/ors/xapp-oai/xapp_bs_connector
   ./run_xapp.sh
   ```

5. **Run xApp Logic** (Python):
   ```bash
   cd /home/w5/ors/xapp-oai/base-xapp
   python slicing_ctrl_influxdb_xapp_kpm.py
   ```

## Configuration Notes

- **gNB E2 Agent**: Already built into compiled `nr-softmodem`
- **gNB Config**: `ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf` includes E2 settings
- **SM Format**: Custom Service Model for RAN slicing (KPM + CCC)
- **Slicing Policy**: JSON format for resource allocation per slice

## Testing Slicing

Once xApp is running:
- Modify `min_ratio` / `max_ratio` in Python logic
- See resource allocation changes in gNB logs
- Monitor throughput per slice via gNB statistics

See `../doc/rrmPolicyJson.patch` for alternative file-based slicing config.
