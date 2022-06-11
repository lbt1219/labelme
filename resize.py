from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=800,height=600):
    img=Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        if new_img.mode == 'P':
            new_img = new_img.convert("RGB")
        if new_img.mode == 'RGBA':
            new_img = new_img.convert("RGB")
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

for jpgfile in glob.glob("/Volumes/My_Passport/LBT/DATASET/wire-dataset/现场拍摄图片/*.jpg"):
    # print(jpgfile)
    convertjpg(jpgfile,"/Volumes/My_Passport/LBT/DATASET/wire-dataset/800600")

