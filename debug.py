from accelerate import Accelerator

accelerator = Accelerator()

print(accelerator.num_processes)