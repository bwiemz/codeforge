import torch
def bytes_to_mib(b): return b / (1024 * 1024)
def bytes_to_gib(b): return b / (1024 * 1024 * 1024)
def sep(title=''):
    w = 80
    if title:
        pad = (w - len(title) - 2) // 2
        eq = chr(61) * pad
        print(eq + ' ' + title + ' ' + eq)
    else:
        print(chr(61) * w)
def reset_mem():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
def peak_mib(): return bytes_to_mib(torch.cuda.max_memory_allocated(0))
def cur_mib(): return bytes_to_mib(torch.cuda.memory_allocated(0))
def gpu_total_mib():
    return bytes_to_mib(torch.cuda.get_device_properties(0).total_mem)
