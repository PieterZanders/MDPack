#!/bin/bash
# Colab: install, then compresstraj chain and cg2all chain (both resumable; results in out/ and out_cg/)
cd /content/bench
D16=1e8rA00,2ikeA00,2jbvA03,3iwcB00,3mzkB03,3nisA01,2p4vA02,4m70B00,1jyoE00,2g36A02,3ogdA01,2lg7A00,1yf2A03,2rsxA00,1r44A00,1b5eA00
DDB=mddb_small_1APQ,mddb_dna_1d89,mddb_rna_2l8f,mddb_protdna_FOXA3,mddb_medium_12CA,mddb_large_1AYX
DCG=mddb_small_1APQ,mddb_protdna_FOXA3,mddb_medium_12CA,mddb_large_1AYX,mddb_membrane_Eprot
(
  pip install -q MDAnalysis pytorch-lightning natsort > ct_install.log 2>&1; pip install -q -e compresstraj --no-deps >> ct_install.log 2>&1
  export EPOCHS=2000
  python colab_compresstraj_driver.py $D16,$DDB c20 > ct_run.log 2>&1
  python colab_compresstraj_driver.py 1e8rA00,3mzkB03,1jyoE00,1b5eA00 l16 >> ct_run.log 2>&1
  echo CT_ALL_DONE >> ct_run.log
) &
(
  cd /content && curl -sL https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 -o mm && chmod +x mm
  export MAMBA_ROOT_PREFIX=/content/mamba
  ./mm create -y -p /content/cgmm -c conda-forge python=3.11 'dgl=2.1.0=cuda118*' 'pytorch=*=cuda118*' mdtraj mdanalysis 'numpy<2' > /content/bench/cg_install.log 2>&1
  P=/content/cgmm/bin/python
  $P -m pip install -q --no-deps git+https://github.com/huhlim/cg2all git+https://github.com/huhlim/SE3Transformer.git 'e3nn>=0.5,<0.6' 'ml-collections<0.2' 'torchdata<0.8' opt_einsum_fx opt_einsum >> /content/bench/cg_install.log 2>&1
  $P -m pip install -q contextlib2 absl-py PyYAML sympy mpmath pydantic >> /content/bench/cg_install.log 2>&1
  $P -m pip install -q --no-deps --force-reinstall git+https://github.com/huhlim/mdtraj.git >> /content/bench/cg_install.log 2>&1
  $P -c 'import cg2all, dgl, torch, e3nn, se3_transformer; print("CGOK", dgl.__version__, torch.__version__, torch.cuda.is_available())' >> /content/bench/cg_install.log 2>&1
  cd /content/bench; export PATH=/content/cgmm/bin:$PATH
  python colab_cg2all_driver.py $D16,$DCG CA_p1000 > cg_run.log 2>&1
  python colab_cg2all_driver.py $D16,$DCG BB_p1000 >> cg_run.log 2>&1
  python colab_cg2all_driver.py $D16,$DCG CA_p100 >> cg_run.log 2>&1
  echo CG_ALL_DONE >> cg_run.log
) &
wait
