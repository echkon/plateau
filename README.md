# README  
### Data for: *Statistical analysis of barren plateaus in variational quantum algorithms*  
**Authors:** Le Bin Ho¹ ², Jesus Urbaneja³, and Sahel Ashhab⁴ ⁵  
¹ Frontier Research Institute for Interdisciplinary Sciences, Tohoku University, Sendai 980-8578, Japan  
² Department of Applied Physics, Graduate School of Engineering, Tohoku University, Sendai 980-8579, Japan  
³ Department of Mechanical and Aerospace Engineering, Tohoku University, Sendai 980-0845, Japan  
⁴ Advanced ICT Research Institute, NICT, Tokyo 184-8795, Japan  
⁵ Research Institute for Science and Technology, Tokyo University of Science, Tokyo 162-8601, Japan   

---

## Overview
This repository contains the numerical data and plotting scripts used in the paper  
*Statistical analysis of barren plateaus in variational quantum algorithms.*  

The study introduces a statistical framework for characterizing barren plateaus (BPs) in variational quantum algorithms (VQAs). Using Gaussian function models and variational quantum eigensolver (VQE) circuits, we identify three types of barren plateaus:  
1. **Localized-dip BPs** – flat landscapes with a sharp dip.  
2. **Localized-gorge BPs** – landscapes with narrow gorge-like structures.  
3. **Everywhere-flat BPs** – uniformly flat landscapes with vanishing gradients.  

The repository provides raw and processed data for the Gaussian analysis, VQE simulations, and genetic algorithm mitigation strategy.  

---

## File Structure  

- **`fig5/`**  
  - Contains data and Python scripts to reproduce Fig. 5 (Gaussian landscapes).  

- **`fig6/`**  
  - Contains data and Python scripts to reproduce Fig. 6 (Gaussian gradients and statistical analysis).  

- **`fig7/`**  
  - Contains data and Python scripts to reproduce Fig. 7 (VQE results for HEA and RPA ansatzes).  

- **`fig8/`**  
  - Contains data and Python scripts to reproduce Fig. 8 (Genetic algorithm mitigation results).  


## Requirements
To reproduce the figures:  
- Python 3.8+  
- NumPy  
- Matplotlib   

Install dependencies with:  
```bash
pip install numpy matplotlib
