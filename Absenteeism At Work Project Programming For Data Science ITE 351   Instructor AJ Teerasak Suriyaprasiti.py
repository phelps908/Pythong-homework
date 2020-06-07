#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION OF THE PROBLEMS
# 
# In this project, I will try to address the absenteeism of workers at a company during work time.
# The problem is that the business department today is highly competive than it used to be in the days and this lead to increased pressure in the work place.
# Therefor, it is reasonable to expect unachievable business goals and elevated risk of becoming unemployed can raised peoples stressed levels. Therefore, the continues presents of such factors can become detromental to a persons health.
# 
# Sometimes this may results to minor illness which of course is not desired. However, it may happened that an employee develop a long term condition an example is depression.
# 
# However, since we will be solving the problems from the point of view of the person in charged of productivity in the company, that means we won't focuse on that aspect of the problem but rather we shall be focusing on predicting absenteeism from work.
# 
# More precisely, we would like to know whether or not an employee is likely to be absent from work for a specific number of hours during work day.
# Having such information at hand can improve decision making by reorganizing the work process in way that will allow us to avoid the lack of productivity and increased the quality of work generated at a work place.
# 
# ## Questions
# 
# 1. How can we defined Absenteeisms at work?
# 
# Answer: absense from work during working hours, resulting in temporary incapacity to execute regular working activity
# 
# 2. Based on what information should we predict whether an employee is expected to be absent or not?
# 
# 3. How would we measure absenteeism?
# 
# However, the purpose of this project is also to explore whether a person presenting certain charateristics is expected to be away from work at some point in time or not. In other words, we would like to know for how many working hours an employee could be away from work based on information such as;
# 
# How far they live from their work place?
# How many children and pets do they have?
# Do they have higher education?
# 
# So these are the basis for these predictions. Now we will need to look at the data at our disposal.
# 
# This data set that i will be working on, is the datasets of an already preexisted problem. This means that this problem could be a reaslistic problem.
# 
# ## Data Prepression phase.
# 
# Regardless of whether you dealing with primary or secondary data, we would have to perform the data preprocessing.
# Data preprocessing is an essential part of every quantitative analysis. It is a group of operations that will covert the raw data into a format that is easier to understand and hence, usefull for further processing and analysis.
# 
# This step will attempt to fix the problems that can be inevitably occur with data gathering.
# Furthermore, it will organize the information in a suitable and practical way before doing the analysis and predictions.
# The data i will be working on is called secondary data otherwise known as raw data.
# 
# This data set was retrieved from www.kaggle.com and here is link of dataset.
# https://www.kaggle.com/nairaminasyan/absenteeism-at-work-data-set
# 
# Attribute Information of the data sets:
# 
# Individual identification (ID)
# 
# Seasons (summer (1), autumn (2), winter (3), spring (4))
# 
# Transportation expense
# 
# Distance from Residence to Work (kilometers)
# 
# Service time
# 
# Age
# 
# Work load Average/day
# 
# Disciplinary failure (yes=1; no=0)
# 
# Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
# 
# Son (number of children)
# 
# Social drinker (yes=1; no=0)
# 
# Social smoker (yes=1; no=0)
# 
# Pet (number of pet)
# 
# Absenteeism time in hours (target) 
# 

# ## Data Preprocessing:
# 
# First, we will preprocess the data. We will devote a significant amount of time to this step as it is a crucial part of every analytical task. 
# 
# We will start working on the ‘Absenteeism_datasets.csv' file and take it to a usable state in a machine learning algorithm.
# 
# ## Machine Learning:
# 
# This section will incorporate the work done in the preprocessing part into the code necessary for making the next step. Namely, to develop a model that will predict the probability of an individual being excessively absent from work.
# 
# In this project, Our model will be a logistic regression model. Numerous machine learning tools and techniques will help us at this stage. At the end, we will store our work as a Python module that we will call ‘absenteeism_module’ and will thus preserve it in a form suitable for further analysis.
# 
# 
# ## Loading the ‘absenteeism_module’:
# 
# In this section we will load the ‘absenteeism_module’ and use its methods to obtain predictions.
# 
# ## Analyzing the predicted outputs in Tableau:
# 
# Finally, we will use Tableau to analyse three separate dependencies between the inputs of our model. The visualizations we will obtain with this software will help us a great deal while looking for insights.
# 

# ### Importing Numpy and Pandas
# ### I will Import Visualization libraries and set %matplotlib inline

# In[1]:


#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from tqdm import tqdm_notebook
from itertools import product
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import HoltWintersResults
from datetime import datetime
import statsmodels.tsa.api as smt
from pylab import rcParams


# ## Now I will Read my CSV file as dataframe called df

# In[2]:


raw_csv_data = pd.read_csv('Absenteeism_datasets.csv')


# ### Now I will be checking the contents of the data set

# In[3]:


raw_csv_data


# Our data sets has 700 rows and 12 colums

# ### Now I will make a copy of the initial dataset.
# 
# Why is this necessary?
# This is because once you start manipulating the dataframe, the changes you make will be applied to the original dataset.
# So therefore by using a copy is playing on the safe side, and to make sure the initial dataset won't be modified.

# In[4]:


# To make a copy of this data set we will run the code
df = raw_csv_data.copy()


# #### Let's check to confirm if this worked

# In[5]:


df


# Yes it worked as can see above which is the exact same copy as the raw csv data

# #### Here we will apply the pandas rows and columns max disploy functions instead of inserting a certin large number as value. 
# #### By using the None it means that pythong will understand this command as  set the no maximum value #
# #### By running these two lines of codes, we have set the prefered display options
# 

# In[6]:


pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[7]:


# To see the dataframe now, We will execute the following code
display(df)


# We can now see that no single row or column is missing
# 
# ### By doing this, we will still have to check for the mission values. This can be done by taking a concise summary of the data frame by executing the following code 

# In[8]:


df.info()


# The output of this shows the number of columns and rows as well as their names and data types.
# From this we can see that our data has no mission values as the columns contains precisely 700 values for each. which is real prof there are no mission values.
# 
# ## Our Analytical approach to be used from now on will be the;
# 
# ### Regression Analysis Approach.
# 
# A popular tool in data analytics, machine learning, advanced statistics, and econometrics, is regression analysis. 
# 
# Roughly speaking, this is an equation which on one side has a variable, called a dependent variable, because its value will depend on the values of all variables you see on the other side.The variables on the right side are all independent, or explanatory. 
# 
# Their role is to explain the value of the dependent variable.
# There are more terms that can be used for these variables in the same context. The dependent variable can also be called a target, while the independent variables can be called predictors.
# 
# Having said that, it is easy to describe what logistic regression is about. It is a type of a regression model whose dependent variable is binary. 
# 
# That is, the latter can assume one of two values – 0 or 1, True or False, Yes or No. Therefore, considering the values of all our features, we want to be able to predict whether the dependent variable will take the value of 0 or 1.
# 
# Apart from logistic regression, there are many other types of equations that can allow you to calculate the dependent variable in a different way. 
# 
# Logistic regression is just one of them – and it is one that has been used massively. 
# 
# Anyway, you would most often hear professionals say that they are trying to find a regression model, or, simply, find a regression, that has a high predictive power. 
# 
# In other words, what they are trying to do is settle upon an equation that could be used to estimate expected values for the dependent variable with great precision. 
# 
# For the moment, this regression analysis and logistic regression will give us the necessary grounds to proceed with the pre-processing part of our task.
# 
# ## Using a statisttical Approach 
# 
# Since our data sets has no missing values is great, this means that we will not have to deal with this.
# So we can use this data for our analysis and hope it produce some great results.
# 
# #### Net what should be do to achieve our task which is to ;
# # Task is to predict the absenteeisms from work.
# 
# Lets break this down a gain into bits. 
# What we do want to predict is absenteeisms from work, therefore we are going to look for a variable in this data set that can represend this phenomenon.
# 
# Second do we have any variable in this data set that can help to predict absenteeisms from work?
# Answer: yes. " Absenteeism Time in Hours" this columns express absenteeism per hour in the data set
# Therefore, in our data set the first instance number 0 tells us that an employee with ID # 11 has been absent for 4hrs on the 7th of July 2015.
# Next instance 1 ID#36 has not been away on the 14th of July 2015 since we see zero hrs of absense.
# Finally. instance 2 with ID# 3 has been away for 2hrs on the 15th of July 2015 and so on.
# 
# All this has to say that the data will eventually tell us whether an employee has been absent for certain significant amount of time on a certain date as has been stored in the absenteeism time in hours colymn.
# 
# So therefore transfering a piece of logic from regression analysis to our example, we could say that absenteeism time in hours could be our dependent variable.
# 
# Finally, what could help us to predict this value for future observations or other columns represent independent variables which could potentially be used in our equetion which could help us to predict whether an individual with a particular characteristics is expected to be absent from work for a certain amount of time or not.
# 
# #### We will start Analizing the data charecteritic by characteristic.
#  We will start by dropping the variables that won't help to assist us in this task.
#  
# ## Drop ID column

# In[9]:


df = df.drop(['ID'], axis = 1)


# In[10]:


df


# ## Reason for Absence

# In[11]:


df['Reason for Absence']


# We most clarify here that, on the left we have indexes - designating the order in which elements appear.
# Starting from 0, naturally.
# 
# Lets now perform check .
# In my DF datasets, reason for absense start from 26, 0, 23 etc
# I will also check to know what the lowest and max vallues are
# 
# ## Reason for Absense MIN and MAX
# 

# In[12]:


df['Reason for Absence'].min()


# In[13]:


df['Reason for Absence'].max()


# Now we want to extra a list containing distict values only.
# However, we want to see the list of different values organized in a list and only shown once.
# To achieve this, we wil use the following code.

# In[14]:


# extract distinct values only
pd.unique(df['Reason for Absence'])


# Here are the required results.
# Another way is by performing the following code.
# 

# In[15]:


df['Reason for Absence'].unique()


# Moreover, the output will tell what kind of values you have in that column which is dtype intergers 64 bits which means they are all of the same type.
# 
# #### Lets also see how many distinct elements are in this conlumn.

# In[16]:


#len() returns the numver of elements in an object
len(df['Reason for Absence'].unique())


# Hence we can concude that the column of interest container 28 different reasons for absence from work
# 
# ###### Since min and max numbers are 0 and 28 it means we should have 29 values.
# ##### This shows we are missing a value.
# ###### We shall use the following code to retrive the missing value which is : sorted() returns a new sorted list from the items in its argument

# In[17]:


sorted(df['Reason for Absence'].unique())


# In this list we can spot that the value we lack is 20
# 
# However, numbers without an inherent meaning are just numbers which means any one can ready them, but not every body can understand them.
# 
# How can we extract some meanings from these numeric  values?
# This implies that, the reasons for absence only column contains intergers only.
# So when you hear a reason number 1, you can't be able to figure out a reason why a person was absent.
# So our question now is what are the 28 reasons which have been substitued with numbers?
# 
# Among some of the reasons given they include: 
# 
# I Certain infectious and parasitic diseases
# II Neoplasms
# III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
# IV Endocrine, nutritional and metabolic diseases
# V Mental and behavioural disorders
# VI Diseases of the nervous system
# VII Diseases of the eye and adnexa
# VIII Diseases of the ear and mastoid process
# IX Diseases of the circulatory system
# X Diseases of the respiratory system
# XI Diseases of the digestive system
# XII Diseases of the skin and subcutaneous tissue
# XIII Diseases of the musculoskeletal system and connective tissue
# XIV Diseases of the genitourinary system
# XV Pregnancy, childbirth and the puerperium
# XVI Certain conditions originating in the perinatal period
# XVII Congenital malformations, deformations and chromosomal abnormalities
# XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
# XIX Injury, poisoning and certain other consequences of external causes
# XX External causes of morbidity and mortality
# XXI Factors influencing health status and contact with health services. 
# 
# And 7 categories without (CID) patient follow-up (22), medical consultation (23), blood donation (24), laboratory examination (25), unjustified absence (26), physiotherapy (27), dental consultation (28).
# 3. Month of absence
# 4. Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
# 5. Seasons (summer (1), autumn (2), winter (3), spring (4))
# 6. Transportation expense
# 7. Distance from Residence to Work (kilometers)
# 8. Service time
# 9. Age
# 10. Work load Average/day
# 11. Hit target
# 12. Disciplinary failure (yes=1; no=0)
# 13. Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
# 14. Son (number of children)
# 15. Social drinker (yes=1; no=0)
# 16. Social smoker (yes=1; no=0)
# 17. Pet (number of pet)
# 18. Weight
# 19. Height
# 20. Body mass index
# 21. Absenteeism time in hours (target) 
# 
# 
# 
# So how do we treat these information analitically similary to the case of the ID column?
# The values here do not have any numeric meanings, but they represent categories that are equally meaningful for example, reason one stance for a certain reason for absense as much as reason 2 stance for another. The fact that in arithmetic terms 2 is greater than 1 has nothing to do with the numbers in the column. From statistical point of view, these numbers are categorical nominal because instead of using numbers from 0 to 28, there could have had names.
# 
# However, from the point of view of database theory, using less characters 1 or 2 digital numbers instead of multiple character strings, will shrink the volume our dataset and thus less data storage will be required to store all information.
# 
# In order to make quantitative analysis, we need to add numeric meaning to our categorical nominal values. And there are various ways to carry out this task. 
# 
# One way is turning these numerical values into dummy variables. However, a dummy variable is an explanatory binary variable that equals 1 if a certain categorical effect is present, and that equals 0  if that same effect is absent.
# 
# So therefore, we would like to have a column where the values of 1 will appear in case an individual had been absent because of reason number and 0 because he has been absent because of another reason.
# 
# From this analysis, we can be sure and certain that an individual has been absent from work because of one, and only one , particular reason.
# 
# We can achieve this goal with the following command.

# ## .get_dummies()
# 
# We will store this output as follows.

# In[18]:


reason_columns = pd.get_dummies(df['Reason for Absence'])


# In[19]:


reason_columns


# We get a seperate data set with 28 columns bearing the numbers from 0 - 28 as their names.
# 
# ## Now we will check whether we have rows with missing values
# 
# The way to achieve this will be to sum all the values in a row and store them in a new which will call it Check and obtain 0, then we have a missing value for the given observation, and if we get 1, then we have a single value along the entire row equal to 1. However, if we see values of 2,3,4 and so on, then we must have either have the value of 1 more tha once or we have had a higher number. Moreoever, we can be certain that an individual has been absent from work because of one, and only one particular reason.
# 
# ## we will do this by using the sum method

# In[20]:


reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns


# So far it has worked correctly as we see the new column "check" with 1 values from top to bottom with no other exceptions.
# 
# #### We will apply the sum method to the check column directly to confirm if this was applied correctly

# In[21]:


reason_columns['check'].sum(axis=0)


# So we got exaply 700 as the len in our df dataframe
# 
# ###### Now we apply the unique method to confirm if all the 700 values are exatly equal to 1.

# In[22]:


reason_columns['check'].unique()


# This confirms that the reasons for absence columns has been continuesly been flowless containing no missing  or incorrect values..
# 
# ##### After this has been confirmed, we shall remove the check cloumn with the use of the drop method
# 

# In[23]:


reason_columns = reason_columns.drop(['check'], axis=1)
reason_columns


#  ### Dropping a Dummy Variable from the Data Set
#  
# The next thing to do in our analysis from a statistical perspective will be to drop the first column of our table.
# 
# In a nutshell, the motivation for us to drop the first column, reason 0, goes like this. If a person has been absent due to reason 0, this means they have been away from work for an unknown reason. Hence, this column acts like the baseline, and all the rest are represented in comparison to this.
# 
# 
# As a consequence, dropping this column would allow us to only conduct the analysis for the reasons we are aware of. And that’s exactly what we want to do - explore whether or not a specific known reason for absence induces an individual to be excessively absent from work. That’s why we don’t really need to keep in our data set information about someone who has been away due to an unknown reason.
# 
# Otherwise, regarding the values stored in the remaining columns, representing reasons 1 and above, we can solidify our rationale by saying the following: Imagine that by default there’s no particular reason for a given individual to be absent from work. If there is any, though, it will be marked with the value of 1 under the corresponding reason number. 
# 
# Therefore, as explained in the article on Quora, to avoid issues with multicollinearity, we must remove one column with dummy variables. To preserve the logic of our analysis, this will be the column that stands for reason 0.
# 
# ## Dropping the unnecessary column

# In[24]:


reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)


# In[25]:


reason_columns


# Perfect this task has been perform successful the columns with zeros no longer there.
# 
# ## We will be Grouping the reasons for absence in this section

# In[26]:


df.columns.values


# The output we retrieve here is the list of columns and their names and it is also important to note that we still have the Reason for Absence column in there.
# 
# So therefore implimenting the same method to reason conlumns we get an output with number of values from 1 to 28 leaving out 20

# In[27]:


reason_columns.columns.values


# ## Three Importantant steps to take to finalize work on this column
# 
# Step 1 if we decide to add all those dummy variables to the current state of df, we will get duplicate information this process is known as multicollonearity and it is something that we should avoid. Therefore, we should drop the reason for absence column from the df table in the usual way.
# 
# #### We will do this with the following code

# In[28]:


df = df.drop(['Reason for Absence'], axis = 1)


# In[29]:


df


# The is done successfully we can't see it in our data frame any more
# 
# 
# 
# ## Grouping the variables
# 
# If we add all these dummy variables in the df dataFrame, we will end up with a dataset containing nearly 40 coloums, this could be too much when dealing with 700 observations.
# 
# In this case we shall consider grouping them.
# This is like reorganizing a certain type of variables into groups in a regression analysis. Thi can also be called classification.
# 
# In our analysis, We will focuse on the qualitative analysis.
# However, in order given variables, we can see that reasons 1 to 14 are all related to diseases that might influence an individual to be absent, this will constitute our first group or class. Our second class will include reasons from 15 to 17 because they are some how related to pregnancy and giving birth. Our group 3 will include reasons 18 to 21 as they are all about poisoning or signs not categorized, and Group 4 will include reasons 22 to 28 which include light reasons such as dental appointment, medical consultation and others.
# This is how we are going to group or classified our data.
# 
# NB after splitting this object into smaller pieces, each piece itself will be a Dataframe object as well.
# 
# These new groups types will called Reason type class number.
# 
# This can be achieved by running the following code.
# 
# We will start by applying the loc method to reason coloumns.
# 
# This is how it works.

# In[30]:


reason_columns.loc[:, '1':'14']


# In[31]:


reason_columns.loc[:, '15':'17']


# In[32]:


reason_columns.loc[:, '18':'21']


# In[33]:


reason_columns.loc[:, '22':'28']


# However, in order to make sure no single individual will have multple reasons for absent.
# First we will be able to achieve this task by substituting our entire row of 14 single values with a new single one in which we would like to obtain 0 if non of the values on the given row were equal to 1.
# 
# Alternatively, we would like to obtain 1 in case somewhere among these 14 columns we have observed the number 1. The former will mean the reasons for absence for this particular individual was not from these particular group of reasons or larter will mean it was.
# 
# We will use a method called .max() to achieve this task. This will return the highest value.
# This can be performed with the following code.

# In[34]:


# here we will illustrate how this told works.
reason_columns.loc[:, '1':'14'].max(axis=1)


# Since the max number is always a single value number, the obtained object will be a pandas series and not data frames.
# Therefore, to create all 4 columns and replicate them as grouped or classified.
# 
# ## Here will run the following code to create the four new columns

# In[35]:


reason_type_1 = reason_columns.loc[:, '1':'14'].max(axis=1)
reason_type_2 = reason_columns.loc[:, '15':'17'].max(axis=1)
reason_type_3 = reason_columns.loc[:, '18':'21'].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)


# ### Our Four Columns have been create.
# ### Lets check them out

# In[36]:


reason_type_1


# In[37]:


reason_type_2


# In[38]:


reason_type_3


# In[39]:


reason_type_4


# #### Everything works just perfectly.
# 
# 
# ## We will Concatenate the Column Values
# 

# In[40]:


df


# We will try to add the newly creted reason type column to this data frame. However, We will use the pandas concatenation function to do the job to the existing df. Naturally, we would like to assign the column names in a more meaningful way, but there are a few ways to do this and here is the one i will use.
# 
# We will create a variable called column_names and will assigned the below list obtaineed to it

# In[41]:


# We will assign this function
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)


# In[42]:


df


# As you can see, the four columns have been added to the right side of our table. Therefore, we are done with analysis for reason for absence. But before we would like to assigned columns to the new created conlumns in a more meaningful way because leaving them as 0-3 doesn't mean anything to any one.
# 
# ## WE will create a new list and assign its elements as a column names of df .
# 
# Here is how we can achieve this. First we will create a new column name called column_names and will assigned the list obtainered below to it. And will rename the last 4 columns as Reason_1 to 4.

# In[43]:


df.columns.values


# In[44]:


column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


# Now we will have to assigned it to columns of our dataframe using the following code.

# In[45]:


df.columns = column_names


# In[46]:


df.head()


# Great the columns names are just same as the ones we have in the column names list

# ## Next task is to reorder the columns of the data frame.
# 
# However, to achieve this will create another column names and call it columns_names_reordered
# 

# In[47]:


column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[48]:


df = df[column_names_reordered]


# In[50]:


df.head()


# Perfect this is what we want
# 
# 
# ### The importance of creating check points
# 
# A checkpoint is an interim save of your work. We do this to have more control of our data.
# What we will do now is create a copy of the current state of the df DataFrame and will use it to tell what state of the data preprocessing we are at.
# 
# 
# ### We will call this process Creating a Checkpoint
# 

# In[58]:


df_reason_mod = df.copy()


# In[59]:


df_reason_mod


# In programming in general, and jupyter in particular, creating checkpoints refers to storing the current version of your code, not really the content of the variables. never the less, the goal remains the same, create a temporary save of your work so that you reduce the risk of losing important data at a later stage
# 
# 
# ### Next on our analysis agender will be the date colum
# 

# # Date

# In[53]:


df_reason_mod['Date']


# Each of its values shows date as follows day of the month / month / year at the same time, forwardshashes are not used in writing integers or floats. So does this column containers strings or values of a different kind?
# We will use the type() function to find the answers.

# In[54]:


type(df_reason_mod['Date'])


# In[55]:


type(df_reason_mod['Date'][0])


# This indicates that it is a python series object.
# However, in 1 column, or in 1 series, we can have values of a sigle data type only!
# These data values have been stored as text. What we are going to do now is introduce a data type called timestamp.
# 
# ### This is a classical data type found in many programming languages out there used for  representing dates and time. This is to convert all values in the date column into time stamp values

# In[60]:


df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')


# In[61]:


df_reason_mod['Date']


# All dates have been converted properly but this time writen in another way. What we see above is the time standard format which is year month day to conclude, We will will use the type() to make sure it remains a series object

# In[63]:


type(df_reason_mod['Date'][0])


# In[64]:


df_reason_mod.info()


# We now have 64 bit date time values in the date column. We will moved on in etracting its month value and day of the week
# 
# ## Extracting the Month Value:
# 
# In order to create the month value column, we must find a way to deduct the month value contained in the df.

# In[65]:


# lets examine the date column
df_reason_mod['Date'][0]


# This is a time stamd 7 of july 2018 and the time is set until midnight and there's no other time. this is because a time stamp created with pandas always contains 2 parts, a date and a time. If time wasn't recorded in our datasets, python will assign 00:00 to the time components authomatically.
# 
# Now to extract only the month, we will use the following code.

# In[66]:


df_reason_mod['Date'][0].month


# Our result is 7 which represents the seventh month of the year which is july which means that months take values from 1 to 12 and not zero to 11.
# 
# We will use the code below to create the month column.
# We would like to create a column value that we will fill with the month values of the date and will assign the values of this list to a new column in df naming it month_value.
# 
# First will assign an empty list before adding values into it.
# 

# In[67]:


list_months = []
list_months


# The output is an empty list.
# 
# We wukk use a loop that will be able to extra the months values the date that we have in every column.
# .append() function will attached the new values obtained from each iteration to the existing content of the designated list.
# range() function this means that for i, taking each of the values from 1 to 700.

# In[68]:


df_reason_mod.shape


# In[69]:


for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)


# In[70]:


list_months


# We see that our list has been altered by years and months starting from July going through December and starting from Janaury all over again.
# 
# By implementing the len() function, we can prove that the list containers as many 700 elements which coincides with the number of records in order data Frame. This means that we are working correctly towards achieving our goal.

# In[71]:


len(list_months)


# Our next tast is creating a new colum in our table and assigning our newly created list of the months using the below code

# In[72]:


df_reason_mod['Month Value'] = list_months


# In[73]:


df_reason_mod.head(20)


# We see that the month value has appeared at the top far right conner of the sscreen.
# What we have achieved now is to help us check whether in specific months of the year an employee turn to be absent more often or not as compared to other months. following this chronology, it may turn out on certain days of the week, workers may be more prone to be away from their desk as compared to other days.
# 
# Our next tast will be to create a column designating the day of the week
# 
# ## Extrating the day of the week column:
# 
# As a data scientist, we won't be expecting to see words in the days of the week column, instead of mondays to sundays, we will have the values as 0 to 6 is so because the week functions will deliver the days of the week like this. " The rules of the game are set by the person who created them"
# 
# To show you how this function works,
# We will use the below code.

# In[74]:


df_reason_mod['Date'][699].weekday()  # This will return an integer corresponding to the day of the week


# After running the cell, we obtain 3 which means that on the 31 of may 2015 was on Thursday

# In[75]:


df_reason_mod['Date'][699]


# To apply a certain type of modification iteratively on each value from a series or a column in a DataFrame, it is a great ide to creare a function that can execute this operation for the one element, and then implement it to all values from the column of interest.
# 
# In order to achieve this task, we will perform the following code

# In[76]:


# we will execute this code to create the function
def date_to_weekday(date_value):
    return date_value.weekday()


# The second part of the process regards to the creation of the week column in df, to this new column, we must assign the value from the date column while implementing the date to week functions to each row. The below code will help to execute the operation we want.

# In[77]:


df_reason_mod['Day of the week'] = df_reason_mod['Date'].apply(date_to_weekday)


# In[78]:


#Lets check the heads of df if this operation was executed correctly
df_reason_mod.head()


# In[79]:


df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod


# Great we see a day of the week column has been attached to the far right conner of our table.
# 
# Now we shall be analyizing the following remaining 5 columns in our datasets. These include the Transportation Expenses, these are cost related to travel expenses like fuel, parking, meals and other charges an employee may reclaim for reinbursement. This means that, transportation is just one of those cost. This table contains monthly transportation expenses of an individual, measured in dollars
# 
# Lets check the data types contained in it.
# 
# by the same token, distance traveled to work contains rounded numbers only, and the values presented in this column are equal to the killometers an individual must travel from home to work. We wwould like to keep these informations in our analysis because it might turns out that distance to work or the time spend traveling might affect the decision of an individual to be absent from work during working hours
# 
# Concerning age, there's not much to be analyse about this variable however, how old a person is, can always have an impact on each or her behaviour.

# In[80]:


type(df_reason_date_mod['Transportation Expense'][0])


# In[81]:


type(df_reason_date_mod['Distance to Work'][0])


# In[82]:


type(df_reason_date_mod['Age'][0])


# In[83]:


type(df_reason_date_mod['Daily Work Load Average'][0])


# This is a float value and represents the daily amount of time a person spends working per day shown in minutes.
# The average amount of time spent working per day, shown in minutes

# In[84]:


type(df_reason_date_mod['Body Mass Index'][0])


# This is an indication for a normal overweight or obese person. However, people who above the norms of their height, often have an additional reason from being absent from work thats why we will include the body mass index our regression analysis.
# only the integers represent the values in this column
# 
# ## Education, Children and Pets
# 
# Is there anything in common about these 3 variables? All these 3 variables represents categorical data containing integers, children and pets indicate how many kids or pets a person has precisely, where as Education is a feature where the numbers do not have any numeric meaning. Therefore, these last 2 columns will remain untouched.
# 
# That means our task will be to transfer the education column into dummy variables.
# We wil achieve this with the help of the map function

# In[86]:


display(df_reason_date_mod)


# In[87]:


df_reason_date_mod['Education'].unique()


# Apply the unique methode to education shows us that this column contain only the values 1 to 4. How can these be interpreted?
# We could say that 1 mean the person has high school education only and does not have any other qualifications
# 2 means that the person has graduated from college and also has a high school degree
# And 3 could stand for post graduate
# and 4 that they could be a master or a doctorate in a given scientific field
# 
# 

# In[88]:


df_reason_date_mod['Education'].value_counts()


# Now using the pandas Value_count method applied to a series objection such as our education column above, allows us to see that the value  1 has been encountered 583 times and 3 73 times 2 40 times and 4 is 4 times. 
# What this information say is that 583 people have only high school qualification only and 40 people gace graduated from college and 73 people have postgraduate qualification while only 4 have a master or doctor.
# 
# This seperating between high school, graduate postgraduate and a master or doctor becomes less relevant for this exercise.
# However, it will make more sense to combine from gradute to a master or doctor together
# 
# In order to achieve this, we will use the below code. tying .map means that we must include a dictionary whose key value pairs will be composed of the existing numbers tht will act as keys and the new numbers which will stand for the values.
# 
# At this point, only high school graduates have been marked with the value of 1 but we want to subsitute this value with a 0 and the rest with a value of 1.
# 
# The below code shall be excuted to achieve this purpose.

# In[89]:


df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})


# In[ ]:


In this 


# In[93]:


df_reason_date_mod['Education'].unique()


# In[92]:


df_reason_date_mod['Education'].value_counts()


# The output obtained is a free that we have worked correctly since the values in education are 0 or 1 only
# 
# 
# ## Final Checkpoint

# In[94]:


df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head(10)


# Conclusion on our data preprocessing with regression techniques. We shall look in our last column which is called Absenteeism Time in Hours and this column don't need need much to be done because  just looking at the list we can see that the first person has been absent for 4hrs and the second person has not been absent at all, also the third person has been accept for 2hrs  while the fourth for 4 hrs and so on.
# 
# There are two reason we would not devote time for the manupulation of this column right now
# 
# 1. The interpretation of this column is straight forward
# 
# 2. Its modification is related to the application of advance statistical techniques in Python

# ###### We will export our data as a a *.csv file and below is the colde to solve this problem with.
# 
# 

# In[ ]:


df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=false)

