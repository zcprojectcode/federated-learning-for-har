# Federated Learning for HAR
This repository investigates the use of a federated learning architecture for privacy-preserving Human Activity Recognition (HAR) to support independent living in smart homes. The framework was designed to fit the Inertial Measurement Unit (IMU) data found here: https://doi.org/10.25919/d7xf-n080. It can be adapted for alternative datasets.     

### Previous work    
Previous studies that evaluated the feasibility of IMU-based HAR using the same dataset can be found here:    
1- Moid Sandhu, et al. "Feasibility of motion sensor-based human activity recognition for supporting independence in smart homes." Maturitas 199 (2025): 108632. https://www.sciencedirect.com/science/article/pii/S0378512225004402    

2- Moid Sandhu, et al. "Fusing IoT Wearable and Object Motion Sensors for Enhanced Activity Recognition in Smart Homes." 2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2025. https://ieeexplore.ieee.org/abstract/document/11253505/    

For details on data collection, please refer here: https://github.com/mmsandhu/IMU-HAR-IL    

#### The structure of this repository is as follows:    

```    
federated-learning-framework    
├── aggregators.py    
├── blocks.py    
├── client_energy.py    
├── client_eval.py    
├── client.py    
├── config.py    
├── data_storage.py    
├── main.py    
├── model.py    
├── plots.py    
├── server.py    
├── utils.py    
├── wearable_energy.py    
```    

### Usage
The federated learning implementation allows different aggregation mechanisms to be compared:        
```
python main.py --algo <algorithm_name>
```
Available algorithm_name options are:     
- fedavg    
- feddane    
- fednova    
- fedper    
- fedprox    
- fedsgd    

config.py can be used to change the system parametrisation. A device selection mechanism has been implemented to improve energy-efficiency and accuracy. Setting smart_selection in config.py to True uses the device selectoin mechanism. Otherwise, random device selection is used. Additional optimisations were also made to further reduce energy usage. Setting reduce_dimensions in config.py to False reduces object sensor dimensions to binary and setting hetero_wearables in config.py to True selects one wearable sensor at a time for each round of sensing.   

Data storage expects data to as a PyTorch tensor. The following tensor structure was used:    
```
checkpoint = {
    "samples":
    "label_map":
    "sensor_index":
}
```

### References:    
The following papers and GitHub repositories were used to develop or inspire this framework. Copyright permissions and additional references have been included where required throughout the repository.     
- FL implementation: https://github.com/frezazadeh/federated-learning    
- Tiny transformer encoder model: Lamaakal, I., Yahyati, C., Maleh, Y. et al. A tiny inertial transformer for human activity recognition via multimodal knowledge distillation and explainable AI. Sci Rep 15, 42335 (2025). https://doi.org/10.1038/s41598-025-26297-2 (Copyright (c) 2025 Ismail Lamaakal)    
- Device selection mechanism formulation: F. Lai, X. Zhu, H. V. Madhyastha, and M. Chowdhury, “Oort: Efficient Federated Learning via Guided Participant Selection,” in 15th USENIX Symposium on Operating Systems Design and Implementation, Santa Clara, CA, (2021) ISBN: 978-1-939133-22-9 AND A. Arouj and A. M. Abdelmoniem, “Towards energy-aware federated learning on battery-powered clients,” in Proceedings of the 1st ACM Workshop on Data Privacy and Federated Learning Technologies for Mobile Edge Network, (2022) https://dl.acm.org/doi/10.1145/3556557.3557952    
- Client device energy usage: H. Cho, A. Mathur, and F. Kawsar, “FLAME: Federated Learning across Multi-device Environments,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, (2022). https://doi.org/10.1145/3550289     
- FedAvg and FedSGD: B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” Proceedings of the
20th International Conference on Artificial Intelligence and Statistics, (2017). https://proceedings.mlr.press/v54/mcmahan17a.html    
- FedDANE: T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smithy, “FedDANE: A Federated Newton-Type Method,” in 53rd Asilomar Conference on Signals, Systems, and Computers, (2019). https://ieeexplore.ieee.org/abstract/document/9049023    
- FedNova: J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization,” in Advances in Neural Information Processing Systems, (2026). https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html    
- FedPer: M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, Federated Learning with Personalization Layers, arXiv:1912.00818 (2019). http://arxiv.org/abs/1912.00818    
- FedProx: T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated Optimization in Heterogeneous Networks,” in Proceedings of Machine Learning and Systems (2020). https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf       
