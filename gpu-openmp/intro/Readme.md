# Device query on LUMI
ROCM provides us with a tool, `romcminfo` to get the device properties. We can get the information of the GPUS that are allocated during a hob via:

```
srun -p dev-g --gpus 1  -n 1  --time=00:30:00 --account=<summerschool_project> rocminfo
```
This will output:

```
Agent 5                  
*******                  
  Name:                    gfx90a                             
  Uuid:                    GPU-8f32185a4cf32f6d               
  Marketing Name:                                             
  Vendor Name:             AMD              
  Device Type:             GPU                                
  Cache Info:              
    L1:                      16(0x10) KB                        
    L2:                      8192(0x2000) KB                    
  Chip ID:                 29704(0x7408)                      
  ASIC Revision:           1(0x1)                             
  Cacheline Size:          64(0x40)                          
  Compute Unit:            110                                
  SIMDs per CU:            4                                  
  Features:                KERNEL_DISPATCH 
  Fast F16 Operation:      TRUE                               
  Wavefront Size:          64(0x40)                           
  Workgroup Max Size:      1024(0x400)                        
  Workgroup Max Size per Dimension:
    x                        1024(0x400)                        
    y                        1024(0x400)                        
    z                        1024(0x400)                        
  Max Waves Per CU:        32(0x20)                           
  Max Work-item Per CU:    2048(0x800)                        
  Grid Max Size:           4294967295(0xffffffff)             
  Grid Max Size per Dimension:
    x                        4294967295(0xffffffff)             
    y                        4294967295(0xffffffff)             
    z                        4294967295(0xffffffff)            
*** Done ***             
```
With `rocm-smi` we can get runtime information:
```
> srun -p dev-g --gpus 1  -n 1 --time=00:30:00 --account=<summerschool_project> rocm-smi

========================= ROCm System Management Interface =========================
=================================== Concise Info ===================================
GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%  
0    50.0c           N/A     800Mhz  1600Mhz  0%   manual  0.0W      0%   0%    
====================================================================================
=============================== End of ROCm SMI Log ================================
```
When a GPU application is running we can seelive info about the temperature, RAM usage, and perctange of GPU used. 
