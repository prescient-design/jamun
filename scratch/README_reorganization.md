# ALA_ALA Swarm Data Reorganization Scripts

## Overview
These scripts reorganize molecular dynamics swarm data from `/data/bucket/vanib/ALA_ALA/swarm_results/` into a machine learning-ready format in `/data2/sules/ALA_ALA_enhanced/`.

## Scripts

### 1. `reorganize_swarm_data.py` - Main Script
**Input Structure:**
- Source: `/data/bucket/vanib/ALA_ALA/swarm_results/`
- 184 directories: `AA_000/`, `AA_001/`, ..., `AA_183/`
- Each contains: `swarm_1ps_001.xtc`, `swarm_1ps_002.xtc`, ..., `swarm_1ps_005.xtc`
- Single PDB file: `/data/bucket/vanib/ALA_ALA/ALA_ALA.pdb`

**Output Structure:**
- Target: `/data2/sules/ALA_ALA_enhanced/`
- `train/` - 172 randomly selected grid codes (860 .xtc + 860 .pdb files)
- `val/` - Remaining 12 grid codes (60 .xtc + 60 .pdb files)

**File Naming Convention:**
- Original: `swarm_1ps_{traj_code}.xtc` → New: `swarm_1ps_{grid_code}_{traj_code}.xtc`
- PDB files: `swarm_1ps_{grid_code}_{traj_code}.pdb` (copied from single source)

**Features:**
- ✅ **Progress bars** with tqdm showing copy progress
- ✅ **Reproducible random split** (seed=42)
- ✅ **mdtraj validation** - Tests that .xtc + .pdb pairs load correctly
- ✅ **Comprehensive logging** - Detailed progress and error reporting
- ✅ **Safe operation** - Only copies, never moves/deletes source data
- ✅ **Verification** - File count validation and structure checking

### 2. `test_reorganize_swarm_data.py` - Test Script
- Creates mock data structure with 5 grid codes
- Tests the reorganization logic with small dataset
- Validates file organization and naming
- Tests mdtraj integration (expected to fail on mock data)
- Verifies progress bar functionality

## Usage

```bash
# Activate conda environment
conda activate jamun

# Navigate to scripts directory
cd scratch

# Run test first (optional)
python test_reorganize_swarm_data.py

# Run reorganization with trajectory split (default)
python reorganize_swarm_data.py trajectory_split

# Or run reorganization with grid split
python reorganize_swarm_data.py grid_split

# Or run without arguments (defaults to trajectory_split)
python reorganize_swarm_data.py
```

## Splitting Strategies

The script supports two different data splitting strategies:

### 1. Grid Split (`grid_split`)
- **Random grid codes split**: 172 grids for train, 12 grids for val, all trajectories
- **Output folder**: `/data2/sules/ALA_ALA_enhanced_full_swarm/`
- **Train**: 172 grid codes × 5 trajectories × 2 file types = 1,720 files
- **Val**: 12 grid codes × 5 trajectories × 2 file types = 120 files

### 2. Trajectory Split (`trajectory_split`) - **DEFAULT**
- **All grids split by trajectory**: trajectories 001-004 for train, 005 for val
- **Output folder**: `/data2/sules/ALA_ALA_enhanced_full_grid/`
- **Train**: 184 grid codes × 4 trajectories × 2 file types = 1,472 files
- **Val**: 184 grid codes × 1 trajectory × 2 file types = 368 files

## Expected Results

**Grid Split structure:**
```
/data2/sules/ALA_ALA_enhanced_full_swarm/
├── train/                           # 1720 files total
│   ├── swarm_1ps_000_001.xtc       # Random 172 grids, all trajectories
│   ├── swarm_1ps_000_001.pdb       
│   └── ... (172 grid codes × 5 trajectories × 2 file types)
└── val/                             # 120 files total  
    ├── swarm_1ps_XXX_001.xtc       # Remaining 12 grids, all trajectories
    └── ... (12 grid codes × 5 trajectories × 2 file types)
```

**Trajectory Split structure:**
```
/data2/sules/ALA_ALA_enhanced_full_grid/
├── train/                           # 1472 files total
│   ├── swarm_1ps_000_001.xtc       # All 184 grids, trajectories 001-004
│   ├── swarm_1ps_000_002.xtc       
│   ├── swarm_1ps_000_003.xtc       
│   ├── swarm_1ps_000_004.xtc       
│   └── ... (184 grid codes × 4 trajectories × 2 file types)
└── val/                             # 368 files total  
    ├── swarm_1ps_000_005.xtc       # All 184 grids, trajectory 005 only
    ├── swarm_1ps_001_005.xtc       
    └── ... (184 grid codes × 1 trajectory × 2 file types)
```

## Dependencies
- **mdtraj** - For trajectory validation (available in jamun environment)
- **tqdm** - For progress bars (available in jamun environment)
- **Standard library** - os, shutil, random, logging, pathlib

## Runtime Estimate
- **Test script**: ~1 second
- **Grid split**: ~52 seconds (1,840 file operations)
- **Trajectory split**: ~37 seconds (1,840 file operations)
- **Disk space needed**: ~Same as source data for each folder (copying, not moving)

## Safety Features
- Never deletes or moves source data
- Validates all paths before starting
- Reports missing files without stopping
- Tests mdtraj compatibility with random samples
- Detailed error logging and recovery 