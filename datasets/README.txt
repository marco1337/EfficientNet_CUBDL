# CUBDL-Related Datasets from the Challenge on Ultrasound Beamforming with Deep Learning

## Details
Details about this database are available in the user agreement that you signed to receive access. Please refer to the agreement for details about usage, restrictions, citation requirements, etc. As a summary, the files in this folder contain:
- 576 image acquisition sequences of ultrasound channel data
- tissue-mimicking phantom manufacturers and model numbers
- tissue-mimicking phantom sound speeds (including reported values, data sheet values, and optimized values after maximizing speckle brightness)
We additionally provide the following in our associated code repository:
- training weights for the two winning networks of the CUBDL challenge

## Citation Requirements
As a reminder, all documents and publications that use any CUBDL-Related Data must acknowledge use of the data and attribute credit to the database by including citations to the following work that made this database possible, as well as a citation to the database itself:

@inproceedings{cubdl_ius,
  title={Challenge on Ultrasound Beamforming with Deep Learning (CUBDL)},
  author={Bell, Muyinatu A Lediju and Huang, Jiaqi and Hyun, Dongwoon and Eldar, Yonina C and van Sloun, Ruud and Mischi, Massimo},
  booktitle={Proceedings of the 2020 IEEE International Ultrasonics Symposium},
  year={2020}
  pages={1-5},
  doi={10.1109/IUS46767.2020.9251434}}
}

@article{cubdl_journal,
  title={Deep Learning for Ultrasound Image Formation: CUBDL Evaluation Framework & Open Datasets},
  author={Hyun, D. and Wiacek, A. and Goudarzi, S. and Rothl{\"u}bbers, S. and Asif, A. and Eickel, K. and Eldar, Y. C. and Huang, J. and Mischi, M. and Rivaz, H. and Sinden, D. and van Sloun, R.J.G. and Strohm, H. and Bell, M. A. L.},
  journal={IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control},
  volume={x},
  number={x},
  pages={xx},
  year={2021},
  publisher={IEEE}
}

@misc{cubdl_data,
doi = {10.21227/f0hn-8f92},
url = {http://dx.doi.org/10.21227/f0hn-8f92},
author = {Bell, Muyinatu A. Lediju and Huang, Jiaqi and Wiacek, Alycen and Gong, Ping and Chen, Shigao and Ramalli, Alessandro and Tortoli, Piero and Luijten, Ben and Mischi, Massimo and Rindal, Ole Marius Hoel and Perrot, Vincent and Liebgott, Hervé and Zhang, Xi and Luo, Jianwen and Oluyemi, Eniola and Ambinder, Emily},
publisher = {IEEE DataPort},
title = {{Challenge on Ultrasound Beamforming with Deep Learning (CUBDL) Datasets}},
url = {https://ieee-dataport.org/competitions/challenge-ultrasound-beamforming-deep-learning-cubdl-datasets}}
year = {2019}
}

## Additional Citation Requirements
In addition to the three required citations listed above, use of any one of the following specific sequence numbers requires the additional citations noted below.

Sequences: JHU001, JHU002
Abbreviated Description: In vivo breast data, acquired with focused ultrasound transmissions
Required Citation: A. Wiacek, O. M. H. Rindal, E. Falomo, K. Myers, K. Fabrega-Foster, S. Harvey, and M. A. Lediju Bell, “Robust short-lag spatial coherence imaging of breast ultrasound data: Initial clinical results,” IEEE Trans- actions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 66, no. 3, pp. 527–540, March 2019.
@article{wiacek2018robust,
  title={Robust short-lag spatial coherence imaging of breast ultrasound data: Initial clinical results},
  author={Wiacek, Alycen and Rindal, Ole Marius Hoel and Falomo, Eniola and Myers, Kelly and Fabrega-Foster, Kelly and Harvey, Susan and Bell, Muyinatu A Lediju},
  journal={IEEE transactions on ultrasonics, ferroelectrics, and frequency control},
  volume={66},
  number={3},
  pages={527--540},
  year={2018},
  publisher={IEEE}
}

Sequences: JHU024, JHU025, JHU026, JHU027, JHU028, JHU029, JHU030, JHU031, JHU032, JHU033, JHU034
Abbreviated Description: In vivo breast data, acquired with plane wave transmissions (Post-CUBDL evaluation data)
Required Citation: Z. Li, A. Wiacek, and M. A. L. Bell, “Beamforming with deep learning from single plane wave RF data,” in 2020 IEEE International Ultrasonics Symposium (IUS). IEEE, 2020, pp. 1–4.
@inproceedings{li2020beamforming,
  title={Beamforming with deep learning from single plane wave RF data},
  author={Li, Zehua and Wiacek, Alycen and Lediju Bell, Muyinatu A.},
  booktitle={2020 IEEE International Ultrasonics Symposium (IUS)},
  pages={1--4},
  year={2020},
  organization={IEEE}
}

Sequences: OSL011, OSL012, OSL013, OSL014
Abbreviated Description: In vivo cardiac data, acquired with focused transmissions
Required Citation: O. M. H. Rindal, S. Aakhus, S. Holm, and A. Austeng, “Hypothesis of improved visualization of microstructures in the interventricular septum with ultrasound and adaptive beamforming,” Ultrasound in Medicine & Biology, vol. 43, no. 10, pp. 2494 – 2499, 2017.
@article{rindal2017hypothesis,
  title={Hypothesis of improved visualization of microstructures in the interventricular septum with ultrasound and adaptive beamforming},
  author={Rindal, Ole Marius Hoel and Aakhus, Svend and Holm, Sverre and Austeng, Andreas},
  journal={Ultrasound in Medicine \& Biology},
  volume={43},
  number={10},
  pages={2494--2499},
  year={2017},
  publisher={Elsevier}
}

Sequences: TSH002, TSH003-TSH501
Abbreviated Description: In vivo brachioradialis data, acquired with plane wave transmissions (500 sequences total)
Required Citation: X. Zhang, J. Li, Q. He, H. Zhang, and J. Luo, “High-quality reconstruc- tion of plane-wave imaging using generative adversarial network,” in 2018 IEEE International Ultrasonics Symposium (IUS), Oct 2018, pp.1–4.
@inproceedings{zhang2018high,
  title={High-quality reconstruction of plane-wave imaging using generative adversarial network},
  author={Zhang, Xi and Li, Jing and He, Qiong and Zhang, Heye and Luo, Jianwen},
  booktitle={2018 IEEE International Ultrasonics Symposium (IUS)},
  pages={1--4},
  year={2018},
  organization={IEEE}
}

Sequence: OSL010
Abbreviated Description: Simulated data that may be used for a dynamic range test
Required Citation: PICMUS, “Plane-wave imaging evaluation framework for medical ultrasound,” 2020. [Online]. Available: https://www.creatis.insa-lyon.fr/EvaluationPlatform/picmus/
@misc{web:picmus,
  author = {PICMUS},
  title = {Plane-wave Imaging evaluation framework for Medical Ultrasound},
year = {2020},
  url = {https://www.creatis.insa-lyon.fr/EvaluationPlatform/picmus/}
}

Sequences: OSL008, OSL009
Abbreviated Description: In vivo carotid artery data, acquired with focused transmissions
Required Citation: A. Rodriguez-Molares, O. M. H. Rindal, J. D’hooge, S.-E. Ma ̊søy, A. Austeng, M. A. L. Bell, and H. Torp, “The generalized contrast- to-noise ratio: a formal definition for lesion detectability,” IEEE Trans- actions on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 67, no. 4, pp. 745–759, 2019.
@article{rodriguez2019generalized,
  title={The generalized contrast-to-noise ratio: a formal definition for lesion detectability},
  author={Rodriguez-Molares, Alfonso and Rindal, Ole Marius Hoel and D’hooge, Jan and M{\aa}s{\o}y, Svein-Erik and Austeng, Andreas and Bell, Muyinatu A Lediju and Torp, Hans},
  journal={IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control},
  volume={67},
  number={4},
  pages={745--759},
  year={2019},
  publisher={IEEE}
}

Sequences: OSL001, OSL015
Abbreviated Description: Simulated data and matching phantom data, acquired with focused transmissions
Required Citation: O. M. H. Rindal, A. Austeng, A. Fatemi, and A. Rodriguez-Molares, “The effect of dynamic range alterations in the estimation of contrast,” IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Con- trol, vol. 66, no. 7, pp. 1198–1208, July 2019.
@article{rindal2019effect,
  title={The effect of dynamic range alterations in the estimation of contrast},
  author={Rindal, Ole Marius Hoel and Austeng, Andreas and Fatemi, Ali and Rodriguez-Molares, Alfonso},
  journal={IEEE transactions on ultrasonics, ferroelectrics, and frequency control},
  volume={66},
  number={7},
  pages={1198--1208},
  year={2019},
  publisher={IEEE}
}

## File Structure
Files are organized within the following folders, named after the descriptions provided in the CUBDL journal paper (i.e., the IEEE TUFFC 2021 citation above):
1_CUBDL_Task1_Data
2_Post_CUBDL_JHU_Breast_Data
3_Additional_CUBDL_Data
For your convenience, each folder contains one or more of the following: 
(1) .hdf5 datafiles listed by the sequence number provided in the Appendix of the CUBDL journal paper. 
(2) .txt files listing any additional citations associated with the data in the corresponding folder. 
(3) Subfolders containing data files, grouped by institution (applies to data that require additional citations). 
(4) The folder named "3_Additional_CUBDL_Data" additionally groups data by acquisition with either plane waves or focused waves. The structure above applies to the contents of the folders within these two additional groupings.

## Conclusion
The Challenge on Ultrasound Beamforming with Deep Learning (CUBDL) was offered as a component of the 2020 IEEE International Ultrasonics Symposium. Recent research effort has been dedicated to developing methods for deep learning in ultrasound imaging, image formation, and beamforming. Despite the clear potential for advantages of deep learning in these applications, there has been a noticeable dearth of evaluation methods that use the same reference data. One major outcome of CUBDL is the provision of a benchmark to compare existing methods using a public dataset and integrated evaluation framework to determine the performance of the deep learning algorithms for image formation algorithms. We also hope that this summary and accessibility of our data, code, and resources will enable future benchmarking and evaluation of newly proposed methods. Visit https://cubdl.jhu.edu for more details.
