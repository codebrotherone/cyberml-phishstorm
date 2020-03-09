# RFC Implementation for PhishStorm

### <i>Detecting Phishing URLs with Streaming Data</i>

This repo will seek to enhance and further efforts made in detecting phishing urls with streaming analytics. 
The goal is to eventually load these models into Google's AI Platform to refine, automate, and serve models using 
realtime streaming analytics. 

The goal of this project is to generate a GAN model architecture that can be deployed to GCP; and automated so that
models are continuously obtaining threat feeds from companies (champion-challenger model succession)
    
Each model competes against each other in a zero sum game that results in a highly performant discriminator model that
can be used to identify phishing attempts. Though more complex models exists (BERT, NLP models, etc...), they are not yet
widely supported/well understood due to the complexity of their architectures. 

![diagram](/research/workflow.png)



##### GAN Architecture
- Generator Roles
    - A ML model that creates fake data.  
    - Data augmentation is provided by the model itself (to be combined with real verified URLs and phishing URLs)
    - Trained using real threat feeds containing legitimate / phishing URLs (domain urls)
    - Trained as a "regression model similar to time-series forecasting, depending on method of feature
    extraction. 
 
- Discriminator Roles
    - Main model that predicts whether a domain url is legitimate or is a phishing attempt.
    - Trained also on real threat feeds containing legitimate / generated phishing URLs.  
    

###### Recurrent neural network w/ LSTM
Create categorical encodings of each URL to generate patterns that the model can detect.
Convert domain URL to one hot encodings for use in an RNN (label encoder, cat encoders, etc...)

##### <i>Random forest classifier</i>

Feed model extracted domain url features (such as Jacquard Similarity Scores) to detect phishing attempts. 
(PhishStorm)

The RFC model seems to be the simplest yet most beneficial model to train. Especially when paired with a generator
in a GAN network. GANs are semi supervised, and alleviate burdens associated with data scarcity while providing
great performance. 


<b>Research Papers</b>
    <ul>https://www.researchgate.net/publication/273169788_PhishStorm_Detecting_Phishing_With_Streaming_Analytics</ul>
    

        

### Features Used from Original Paper
    ![src=research/urlset-features.png]