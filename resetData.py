import glob
import os
low_dir = ".\\LOLdataset\\our485\\low\\"
high_dir = ".\\LOLdataset\\our485\\high\\"
low_fl = glob.glob(low_dir+"*.png")
high_fl = glob.glob(high_dir+"*.png")
idx = 1
for name in high_fl:
    newname = high_dir+str(idx)+".png"
    os.rename(name,newname)
    idx=idx+1
idx = 1
for name in low_fl:
    newname = low_dir+str(idx)+".png"
    os.rename(name,newname)
    idx=idx+1
