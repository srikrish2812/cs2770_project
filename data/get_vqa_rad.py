from huggingface_hub import snapshot_download
snapshot_download(
        repo_id = 'flaviagiammarino/vqa-rad',
        repo_type='dataset',
        local_dir = '/ix/cs2770_2026s/abn80/cs2770_project/data/vqa-rad')
