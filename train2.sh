# python train.py --algo hasac --env lag --exp_name run_formal --task '2v2/ShootMissile/HierarchyVsBaseline'
# python train.py --algo hasac --env lag --exp_name run_formal --task '2v2/NoWeapon/vsBaseline'
python train.py --algo hasac --env lag --exp_name run_formal --task '2v2/NoWeapon/Selfplay'
python train.py --algo hasac --env lag --exp_name run_formal --task '2v2/NoWeapon/HierarchySelfplay'
python train.py --algo hasac --env lag --exp_name run_formal --task '2v2/ShootMissile/HierarchySelfplay'

