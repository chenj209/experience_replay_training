import torch
import subprocess

def get_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,pci.bus_id', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e.stderr}")
        return []

def print_gpu_ids():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    nvidia_smi_output = get_nvidia_smi_output()
    if not nvidia_smi_output:
        print("Failed to retrieve nvidia-smi output")
        return

    gpu_info = {line.split(',')[2].strip(): (line.split(',')[0].strip(), line.split(',')[1].strip()) for line in nvidia_smi_output}

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        pci_bus_id = torch.cuda.get_device_properties(i).pci_bus_id
        pci_device_id = torch.cuda.get_device_properties(i).pci_device_id
        gpu_key = f"{pci_bus_id:02x}:{pci_device_id:02x}"
        if gpu_key in gpu_info:
            gpu_index, gpu_name = gpu_info[gpu_key]
            print(f"GPU {i}: {gpu_name} (nvidia-smi ID: {gpu_index})")
        else:
            print(f"GPU {i}: Unknown (PCI Bus ID: {gpu_key})")

if __name__ == "__main__":
    print_gpu_ids()
