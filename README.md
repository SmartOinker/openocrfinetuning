OpenOCR may be hard to figure out when you wanna finetune on custom dataset. I present the process of finetuning SVTRv2_mobile (from OpenOCR) on my custom dataset.

Run this
```
git clone https://github.com/SmartOinker/openocrfinetuning
cd openocrfinetuning/
conda create -n ocr python=3.13
conda activate ocr
pip install -r requirements.txt
```
Before installing requirements.txt, open it and check if `https://download.pytorch.org/whl/cu128` is suitable for your system (i.e your system should be using cuda 128)

To start finetuning, look at `dataset_openocr` for the format of the dataset, and prepare your own accordingly. Then look at `output/config.yml` for modifying configs to suit your usecase (e.g. shorter `max_text_length` for word level instead of line level recognition). Then run 
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 tools/train_rec.py --c ./output/config.yml
```
Results will be in the `output/` folder. `output/best.pth` is a model trained from scratch on my proprietary dataset, you can finetune from it.

After finetuning, look at `0-main.ipynb` for inference code. 

Refer to `1-sidequest.ipynb` for my YOLO line detection model. In my usecase, the line detection model is used to extract lines images from scans, before the OpenOCR recognition model performs OCR on the line images. Refer to `1-sidequest.ipynb` for training the YOLO line detection model, and `./dataset_yolo` for dataset format required for YOLO line detection model training (standard YOLO dataset)

Lastly, refer to `2-integrating_sidequest.ipynb` to see how my OpenOCR recognition model works with my YOLO line detection model. My methodology could serve as an inspiration for your OCR finetuning endeavours. 

*Dataset uploaded to the repo is just a sample, and is far too little for training. Please prep your own dataset if you want good results.

