import numpy as np
import einops

a=np.arange(6)
b=einops.rearrange(a,'(a b) -> (b a)',a=3)
print(b)