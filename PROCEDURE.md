
# Pregled označenega dataseta

- lokalno naredi conda environment: 
    - `conda env create -f ~/Diploma-GECO2-with-Depth-information/GECO2/environment.yml`
    - `conda activate geco2`
    - `cd Diploma-GECO2-with-Depth-information/GECO2`
    - `pip install gradio_image_prompter-0.1.0-py3-none-any.whl`
    - `pip install opencv-python`
- predpostavimo da je struktura map:
    ```
    ./GECO2
        /IOCfish5kDataset
            /divided
                /n
                    /color
                    /images
                    /xml
                ...
            ...
        ...
    ```
    kjer imamo v mapaj color globinske slike, images RGB slike in xml oznake
- zaženi: `python IOCfish5kDataset/manual_annotations.py --usedivided /home/erik/Diploma-GECO2-with-Depth-information/GECO2/IOCfish5kDataset/divided/n`


# Začetek

- nastavi si HPC dostop in preglej ukaze (glej SLING_README.md)
- pridobi informacije o HPC okolju:
    - prijava `ssh er52565@hpc-login.arnes.si -i ~/.ssh/id_ed25519_SLING`, nato zaženi na HPC login nodu
    - poglej svoj username in domač direktorij: `whoami && echo $HOME` (er52565 in /d/hpc/home/er52565)
    - preglej GPU particije in stanja vozlišč: `sinfo --Node --long` in prekopiraj v `hpc_train.sh` ter `hpc_inference.sh` pod `#SBATCH --partition=...`
    - preglej vse rezervacije: `sinfo --reservation`
    - poglej razpoložljive Anaconda module: `module avail Anaconda 2>&1 | grep -i anaconda` in prekopiraj v `hpc_train.sh` ter `hpc_inference.sh` pod `module load ...`
    - razpoložljive CUDA verzije: `module avail CUDA 2>&1 | grep "CUDA/"` in prekopiraj v `hpc_train.sh` ter `hpc_inference.sh` pod `module load ...`
    - preveri ali cnt2 okolje že obstaja: `module load Anaconda3 && conda env list`
    - preglej prostor na disku: `quota -s`
- nastavi si SSH dostop: glej `/home/erik/Diploma-GECO2-with-Depth-information/_SLING/SLING_README.md`
- prenesi kodo in dataset na HPC:
    - na HPC poženi (spremeni glede na svoj username in home directory!): `mkdir -p /d/hpc/home/er52565/GECO2/models /d/hpc/home/er52565/GECO2/results /d/hpc/home/er52565/GECO2/logs`
    - na lokalnem računalniku zaženi:
        - prenesi kodo projekta: `rsync -avP --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='sam2/build' --exclude='build/' --exclude='IOCfish5kDataset/' ~/Diploma-GECO2-with-Depth-information/GECO2/ er52565@hpc-login.arnes.si:/d/hpc/home/er52565/GECO2/`
        - prenesi skripte za Slurm: `rsync -avP ~/Diploma-GECO2-with-Depth-information/_SLING/ er52565@hpc-login.arnes.si:/d/hpc/home/er52565/_SLING/`
        - prenesi IOCfish5k dataset (samo enkrat - traja dlje): 
            - uporabi `python prepare_done_images.py --folder /home/erik/Diploma-GECO2-with-Depth-information/GECO2/IOCfish5kDataset/divided/2300` da pridobiš novo mapo s samo podatki že preverjenih slik:
                `python prepare_done_images.py --folder /home/erik/Diploma-GECO2-with-Depth-information/GECO2/IOCfish5kDataset/divided/2300`
            - dobiš mapo `/home/erik/Diploma-GECO2-with-Depth-information/GECO2/IOCfish5kDataset/divided/2300_done`
            - prenesi mapo na HPC: `rsync -avP --progress --mkpath ~/Diploma-GECO2-with-Depth-information/GECO2/IOCfish5kDataset/divided/2300 er52565@hpc-login.arnes.si:/d/hpc/home/er52565/GECO2/IOCfish5kDataset/divided/2300_done`
- poveži git in GitHub repozitorij:
    - `cd /d/hpc/home/er52565/GECO2`
    - `git init`
    - `git remote add origin https://github.com/radov02/GECO2.git`
    - check: `git remote -v`
- vzpostavi conda okolje `cnt2` na HPC login nodu:
    - `module load Anaconda3`
    - `conda env create -f /d/hpc/home/er52565/GECO2/environment.yml -n cnt2`
    - `module load Anaconda3 && eval "$(conda shell.bash hook)" && conda activate cnt2`
    - `pip install -e /d/hpc/home/er52565/GECO2/sam2`
    - `pip install torchaudio==2.11.0`
    - `pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`
    - `pip install torchvision`
- zgradi Deformable-DETR CUDA razširitev v conda okolje - potrebno narediti na računskem vozlišču:
    - zahtevaj interaktivno sejo z GPU: `salloc --partition=gpu --gres=gpu:1 --time=00:30:00 --ntasks=1`
    - ko dobiš vozlišče (`salloc: Granted job allocation 66069525; salloc: Nodes gwn10 are ready for job`): 
        - zaženi bash na računskem vozlišču: `srun --jobid=66069525 --pty bash`
        - `module load Anaconda3 && module load CUDA/12.3.0 && eval "$(conda shell.bash hook)" && conda activate cnt2`
        - `cd /d/hpc/home/er52565/GECO2/Deformable-DETR/models/ops`
        - `python setup.py build install`
        - `exit`
        - in še releasaj resource allocation za job:
            - `scancel <JOB_ID>`
            - in preveri: `squeue --job=<JOB_ID>`




# Treniranje
- prek SSH se prijavi na HPC: `ssh er52565@hpc-login.arnes.si -i ~/.ssh/id_ed25519_SLING`
- `cd /d/hpc/home/er52565/GECO2`
- `git pull`
- lokalno zaženi: `rsync -avP ~/Diploma-GECO2-with-Depth-information/_SLING/ er52565@hpc-login.arnes.si:/d/hpc/home/er52565/_SLING/`
- preglej in uredi parametre za treniranje:
    - `nano /d/hpc/home/er52565/_SLING/hpc_train.sh`
- zaženi posel:
    - `mkdir -p /d/hpc/home/er52565/GECO2/results /d/hpc/home/er52565/GECO2/models`
    - `sbatch /d/hpc/home/er52565/_SLING/hpc_train.sh`
- spremljaj potek treniranja:
    - stanje posla v čakalni vrsti: `squeue --user=er52565` ali bolje: `watch -n 1 "squeue --partition=gpu | head -n 35"`
    - log v živo: `tail -f /d/hpc/home/er52565/GECO2/results/train_<SLURM_JOB_ID>.out`
    - podrobnosti o poslu (stanje, vozlišče, čas): `watch -n 1 scontrol show job <SLURM_JOB_ID>`
    - (pregled zaključenih poslov zadnjih 3 dni: `sacct --starttime $(date -d '3 day ago' +%D-%R) --format JobID,JobName,Elapsed,State,ExitCode`)
    - prekini posel po potrebi: `scancel <SLURM_JOB_ID>`
- prenesi uteži modela na lokalni računalnik:
    - lokalno zaženi `rsync -avP er52565@hpc-login.arnes.si:/d/hpc/home/er52565/GECO2/models/ ~/Diploma-GECO2-with-Depth-information/GECO2/models/`




# Inference
- prek SSH se prijavi na HPC: `ssh er52565@hpc-login.arnes.si -i ~/.ssh/id_ed25519_SLING`
- `cd /d/hpc/home/er52565/GECO2`
- `git pull`
- lokalno zaženi: `rsync -avP ~/Diploma-GECO2-with-Depth-information/_SLING/ er52565@hpc-login.arnes.si:/d/hpc/home/er52565/_SLING/`
- preglej in uredi parametre za inferenco:
    - `nano /d/hpc/home/er52565/_SLING/hpc_inference.sh`
    - pazi da ima spremenljivka `MODEL_NAME` enako vrednost kot je ime checkpointa v `models/` (brez `.pth`)
- zaženi inferenco na HPC nodu:
    - `mkdir -p /d/hpc/home/er52565/GECO2/logs`
    - `sbatch /d/hpc/home/er52565/_SLING/hpc_inference.sh`
- preveri log in rezultate:
    - log v živo: `tail -f /d/hpc/home/er52565/GECO2/logs/<SLURM_JOB_ID>.out`
    - rezultati ko je posel zaključen: 
        - `ls /d/hpc/home/er52565/GECO2/models/GECO2_IOCfish_visuals/`
        - `cat /d/hpc/home/er52565/GECO2/models/GECO2_IOCfish_test_results.txt`
- prenesi rezultate na lokalni računalnik:
    - lokalno poženi:
        - `mkdir -p ~/Diploma-GECO2-with-Depth-information/results/visuals`
        - `rsync -avP er52565@hpc-login.arnes.si:/d/hpc/home/er52565/GECO2/models/GECO2_IOCfish_visuals/ ~/Diploma-GECO2-with-Depth-information/results/visuals/`
        - `rsync -avP er52565@hpc-login.arnes.si:/d/hpc/home/er52565/GECO2/models/GECO2_IOCfish_test_results.txt ~/Diploma-GECO2-with-Depth-information/results/`




# Lokalno testiranje z demo_gradio.py
- če še nisi, pridobi uteži modela iz HPC: `rsync -avP er52565@hpc-login.arnes.si:/d/hpc/home/er52565/GECO2/models/ ~/Diploma-GECO2-with-Depth-information/GECO2/models/`
- nastavi lokalno okolje:
    - `conda activate geco2` (lokalno okolje iz koraka "Pregled označenega dataseta")
    - `pip install opencv-python gradio`
    - `pip install /home/erik/Diploma-GECO2-with-Depth-information/GECO2/gradio_image_prompter-0.1.0-py3-none-any.whl`
- zaženi demo:
    ```
    cd ~/Diploma-GECO2-with-Depth-information/GECO2 && python demo_gradio.py \
    --model_name GECO2_IOCfish \
    --model_path ./models \
    --backbone resnet50 \
    --reduction 16 \
    --image_size 1024 \
    --emb_dim 256 \
    --num_heads 8 \
    --kernel_dim 1 \
    --num_objects 3
    ```
- odpri brskalnik na `http://localhost:7860`
- v `demo_gradio.py` nastavi pot do uteži:
    ```python
    model.load_state_dict(torch.load('path/to/modelName.pth', weights_only=True)['model'], strict=False)
    ```
- faq:
    - deluje brez GPU, le počaseje
    - ne potrebuje Deformable-DETR CUDA razširitve, saj direktno kliče `counter_infer.build_model`, ki ne vsebuje deformabilne pozornosti