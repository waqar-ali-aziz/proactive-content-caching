## Project: Federated Transfer Learning-Based Proactive Collaborative Content Caching in Cellular Networks

In the early stages of the internet, content was primarily created by a small number of prominent content service providers. Nowadays, many small-scale and individual content providers exist. These content providers present new challenges for Content Delivery Networks (CDNs). One solution is to bring content closer to the end-users, reducing the repetitive transmission of the same content over backbone networks. However, edge resources are limited, and reactive caching methods like first-in-first-out, least recently used, and least frequently used struggle to manage cellular cache resources effectively. 

Predictive caching methods offer a promising solution to address this problem. These predictive methods employ machine learning models to predict the relevance of content by forecasting its popularity. Nevertheless, the vast cellular dataset poses challenges for machine learning models. Federated Transfer Learning techniques have been applied to reduce model training time by training the model across multiple base stations. Still, there is room for improvement in adapting federated transfer learning to the nature of cellular traffic.

This project aims to enhance the utilization of cellular cache resources by predicting the popularity rank of content. Its primary objective is to improve the efficiency of the predictive model by optimizing the federated transfer learning technique in cellular networks.

## Content of the code
The "translation file" code is utilized to translate the Chinese dataset into the English language. The original Chinese language dataset is accessible at https://github.com/lichenyu/Datasets/tree/master/Youku_Popularity_151212_151221.

The "optimized federated transfer learning" code is designed to enhance the efficiency of the federated transfer learning technique.

The "cache optimization problem" code is employed to address the optimization problem, specifically a Binary Integer Problem (BIP) in this case, which is solved using the Balas Additive Algorithm (BAA). The interested readers can read the paper "An additive algorithm for solving linear programs with zero-one variables" to get more knowledge about BAA. 
