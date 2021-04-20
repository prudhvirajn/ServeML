# ServeML
<b>UC CS 21 Senior Design Project: "Machine Learning and Deep Learning: What Happens After Training?"</b>


This is a research project investigating how machine learning systems behave once they have been deployed. For now, we are researching how do these neural networks trained for classification behave in response to augmentations. 

<em>Project done by Prudhviraj Naidu.</em><br>
<em>Advised by Dr. Nan Niu.</em>

## Table of Contents
1. [Project Description][1]
2. [User Interface Specification][2]
3. [Test Plan and Results][3]
4. [User Manual][4]
5. [Spring Final PPT Presentation][5]
6. [Final Expo Poster][6]
7. Assessments
    - [Initial Self-Assessments][7]
    - [Final Self-Assessments][8]
8. [Summary of Hours and Justification][9]

[1]: https://github.com/prudhvirajn/ServeML/blob/master/Project_Reports/Project-Description.md
[2]: https://github.com/prudhvirajn/ServeML/blob/master/Project_Reports/UI_Specifications.md
[3]: https://github.com/prudhvirajn/ServeML/blob/master/Project_Reports/Essays/Test_Plan_and_Results.pdf
[4]: https://github.com/prudhvirajn/ServeML/blob/master/UserDocs.md
[5]: https://docs.google.com/presentation/d/1p9VyNKzG8q-hKEUNTtu8iuidSMYgR7-X44fjId_4eGQ/edit?usp=sharing
[6]: https://drive.google.com/file/d/1m5TOmc0FHDoSsKAn4NPI9fAdFD_rz5R1/view?usp=sharing
[7]: https://github.com/prudhvirajn/ServeML/blob/master/Project_Reports/Essays/Initial_Assessment_Prudhviraj_Naidu.pdf
[8]: https://github.com/prudhvirajn/ServeML/blob/master/Project_Reports/Essays/Final_Assessment_Prudhviraj_Naidu.pdf
[9]: https://github.com/prudhvirajn/ServeML/blob/master/Project_Reports/Appendix/Summary_Hours_Prudhviraj_Naidu.md

## FAQ

Q1: How to download the dataset?

Ans: Download the entire ImageNet dataset and use the specific wnids provided in the text files in the data directory.

Q2: How long does it take to train?

Ans: It took 8 hours to train for 120 epochs on Kaggle GPU

Q3: Can we use TPU?

Ans: No, the keras training pipeline uses data augmentation operations which are currently not supported by TPU. 
