### Capstone 

Intrusion Detection on a host computer with ML/AI detection

Author: Andres Garcia

#### Executive summary

#### Rationale
Why should anyone care about this question?

The ML/AI instrusion detection system is needed to detect malicious attacks, un-intended installations, remote access to IP information, etc. 
The ML/AI system can have several algorithms suited for different scenarios, but needs to be fast and efficient. 

#### Research Question
The question right now is what process and algorithms will be used. Several classifactions were used, but some failed to run due to GPU requirements
and also the json scripts needed to test the runs. Postman was used to test the HTML and JSON scripts, but this was not sufficient. 

What tools will be used is the ither question? VSCode, StreamLit, Pickle, Docker, and others. There were several issues along the way that have to 
be trouble shooted and resolved.

#### Data Sources
What data will you use to answer you question?

The data needed will be the score, accuracy, runtimes, and portability to run on the frontend of an HTML webpage which causes its own issues.

#### Methodology
What methods are you using to answer the question?
Right now several different tutorials are being taken. 
1. Streamlit
2. YouTube videos on Neural Networks
3. Courses on Neural Network applications and code that works
4. Several IDE's or platforms that work from prototype to production. 
5. Keep trying diffrent type of json, javascript, and python methods.


#### Results
What did your research find?

We found that tensorflow has its own flaws. It often complains that GPUs are not present and will not run. You are almost forced to use the free GPU 
resources from COLAB. We tried to SSH the resources from COLAB while using Cloudflare for SSH Tunnel security, but did not have sufficient time to 
work out the bugs and configuration settings.

#### Outline of project


This project has four major parts :

    model.py - This contains code fot our Machine Learning model to predict cybersecurity issue.
    app.py - This contains Flask APIs that receives employee details through GUI or API calls, used the pickle model, and predicts the 
		probability of an attack with root login failures.
    index.html- This folder contains the HTML file to allow user to enter the number of times of failed root logins.

    myModel.pkl - The optimized model. This had issues when being created due to software module versions in conflict with each other.
    

Running the project

    Ensure that you are in the project home directory. Create the machine learning model by running below command from command prompt -

			python model.py

This would create a serialized version of our model into a file myModel.pkl

    		Run app.py using below command to start Flask API

			python app.py

By default, flask will run on port 5000.

    Navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000


##### Contact and Further Information

andy.garcia.usa@gmail.com



