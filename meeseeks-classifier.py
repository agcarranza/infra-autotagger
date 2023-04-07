
# coding: utf-8

# In[1]:


import re
import ast
import json
import random
import pickle
import datetime
import unicodedata
import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt

import prestodb
from getpass import getpass
from linkedin.jiraclient import JIRA

import os
os.environ["WANDB_DISABLED"] = "true"


# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; }</style>"))
pd.set_option('display.max_colwidth', 100)


# In[3]:


from transformers.file_utils import is_tf_available, is_torch_available

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

set_seed(1)


# # Load Data

# ## Methods to extract data

# In[4]:


"""
Class containing methods that load u_meeseeks tables via presto
"""
class PrestoConnection(object):
    
    def __init__(self):
        self.user = input("LDAP user name:")
        self.password = getpass("LDAP password + 2FA:")
        self.conn = prestodb.dbapi.connect(
            host = 'presto-obfuscated.grid.linkedin.com',
            port = 8443,
            user = self.user,
            catalog = 'hive',
            schema = 'default',
            http_scheme = 'https',
            auth = prestodb.auth.BasicAuthentication(self.user, self.password),
        )
        
    def get_incident_checks(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.incident_checks
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        incident_checks = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return incident_checks
    
    def get_incident_group_matches(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.incident_group_matches
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        incident_group_matches = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return incident_group_matches
    
    def get_incident_groups(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.incident_groups
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        incident_groups = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return incident_groups
    
    def get_incidents(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.incidents
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        incidents = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return incidents
    
    def get_iris_metadata(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.iris_metadata
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        iris_metadata = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return iris_metadata
    
    def get_iris_steps(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.iris_steps
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        iris_steps = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return iris_steps
    
    def get_jira_group_matches(self, start_date, end_date):
        sql = f"""
            SELECT
                *
            FROM u_meeseeks.jira_group_matches
            WHERE
                datepartition BETWEEN '{start_date}' AND '{end_date}'
            """
        cur = self.conn.cursor()
        cur.execute(sql)
        jira_group_matches = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        return jira_group_matches
    
    def get_all_tables(self, start_date, end_date):
        incident_checks = self.get_incident_checks(start_date, end_date)
        incident_group_matches = self.get_incident_group_matches(start_date, end_date)
        incident_groups = self.get_incident_groups(start_date, end_date)
        incidents = self.get_incidents(start_date, end_date)
        iris_metadata = self.get_iris_metadata(start_date, end_date)
        iris_steps = self.get_iris_steps(start_date, end_date)
        jira_group_matches = self.get_jira_group_matches(start_date, end_date)
        
        dfs = {'incident_checks':incident_checks,
              'incident_group_matches':incident_group_matches,
              'incident_groups':incident_groups,
              'incidents':incidents,
              'iris_metadata':iris_metadata,
              'iris_steps':iris_steps,
              'jira_group_matches':jira_group_matches}
        
        return dfs


# In[5]:


"""
Class containing methods that load Jira tickets data
"""
class JiraConnection(object):
    
    def __init__(self):
        self.user = input("LDAP user name:")
        self.password = getpass("LDAP password:")
        self.jira_client = JIRA(user=self.user, password=self.password)

    def _extract_jira_values(self, jira_entries):
        return_list = []
        if jira_entries:
            for value in jira_entries:
                return_list.append(value["value"].strip())
        return return_list

    def _extract_jira_comments(self, jira_comments):
        return_list = []
        if jira_comments:
            for jira_comment in jira_comments:
                author = jira_comment["author"]["name"]
                update_author = jira_comment.get("updateAuthor", {}).get("name", "")
                created = jira_comment["created"]
                updated = jira_comment.get("updated", None)
                comment = jira_comment["body"]
                return_list.append({"author":author,
                                    "update_author":update_author,
                                    "created":created,
                                    "updated":updated,
                                    "comment":comment})
        return return_list

    def _jira_issue_to_dict(self, jira_issue):
        """
        Takes a given jira issue and converts it to a dictionary
        :param jira_issue: jira issue returned by a jira client lookup
        :return: dictionary containing information from Jira
        """
        jira_ticket_id = jira_issue.key
        issue_fields = jira_issue.raw["fields"]

        issue_type = issue_fields.get("issuetype", {}).get("name", "")
        issue_labels = issue_fields.get("labels", [])

        severity = issue_fields.get("Severity", {}).get("value", "")
        user_impact_data = issue_fields.get("User_Impact", "")
        user_impact = user_impact_data if user_impact_data else ""
        impact = issue_fields.get("Impact", {}).get("value", "")

        products_impacted = self._extract_jira_values(issue_fields.get("customfield_14201", []))
        devices_impacted = self._extract_jira_values(issue_fields.get("customfield_14202", []))
        fabrics_impacted = self._extract_jira_values(issue_fields.get("customfield_14203", []))
        impacted_teams = self._extract_jira_values(issue_fields.get("customfield_10356", []))
        responsible_teams = self._extract_jira_values(issue_fields.get("customfield_10570", []))
        responsible_services = self._extract_jira_values(issue_fields.get("customfield_17001", []))
        issue_detected_via = self._extract_jira_values(issue_fields.get("Issue_Detected_Via", []))

        summary = issue_fields.get("summary", "")
        description = issue_fields.get("description", "")
        domain_data = issue_fields.get("Domain", {})
        domain = domain_data.get("value", "") if domain_data else ""

        issue_start_time = issue_fields.get("Start_Time", None)
        issue_detection_time = issue_fields.get("Issue_Detection_Time", None)
        issue_mitigation_time = issue_fields.get("Issue_Mitigation_Time", None)
        issue_resolution_time = issue_fields.get("Issue_Resolution_Time", None)

        jira_comments = self._extract_jira_comments(issue_fields.get("comment", {}).get("comments", []))

        root_cause = issue_fields.get("Root_Cause", "")
        # its possible for this field to exist but to be None, hence the extra guarding
        root_cause_category_data = issue_fields.get("Root_Cause_Category", {})
        root_cause_category = root_cause_category_data.get("value", "") if root_cause_category_data else ""

        jira_root_cause_mapping = {"Code Change / Full deployment" : "Deploy",
                                   "Configuration" : "Deploy - Config",
                                   "Capacity" : "Capacity",
                                   "Infrastructure Failure" : "Infrastructure",
                                   "Hardware Failure" : "Hardware",
                                   "Single Node Failure" : "Outlier Bad Hosts",
                                   "Code Change" : "Deploy",
                                   "CM" : "CM",
                                   "Code Change / Canary" : "Deploy",
                                   "LIX / Feature Ramp" : "LIX",
                                   "Bug" : "Deploy"}

        jira_label = jira_root_cause_mapping.get(root_cause_category, "")
            
        return {"jira_ticket_id": jira_ticket_id,
                "jira_label": jira_label,
                "root_cause_category": root_cause_category,
                "root_cause": root_cause,
                "issue_type": issue_type,
                "issue_labels" : issue_labels,
                "severity": severity,
                "user_impact": user_impact,
                "impact": impact,
                "products_impacted": products_impacted,
                "devices_impacted": devices_impacted,
                "fabrics_impacted": fabrics_impacted,
                "impacted_teams": impacted_teams,
                "responsible_teams": responsible_teams,
                "responsible_services": responsible_services,
                "issue_detected_via": issue_detected_via,
                "summary": summary,
                "description": description,
                "domain": domain,
                "jira_comments" : jira_comments}

    def get_jira_tickets(self, jira_group_matches):
        tickets = []
        for jira_ticket_id in jira_group_matches["jira_ticket_id"].unique():
            try:
                jira_issue = self.jira_client.issue(jira_ticket_id)
                jira_dict = self._jira_issue_to_dict(jira_issue)
                tickets.append(jira_dict)
            except:
                print("Jira ticket", jira_ticket_id, "not found.")  

        return tickets
    
    def get_jira_text(self, jira_group_matches):
        tickets = self.get_jira_tickets(jira_group_matches)
        
        jira_comments = []
        for ticket in tickets:
            text = ""        
            if ticket["root_cause"]:
                text += ticket["root_cause"] + " "
            if ticket["summary"]:
                text += ticket["summary"] + " "
            if ticket["description"]:
                text += ticket["description"] + " "
            for comment in ticket["jira_comments"]:
                text += comment["comment"] + " "

            data = {"jira_ticket_id":ticket["jira_ticket_id"], "jira_label":ticket["jira_label"], "jira_text":text}
            jira_comments.append(data)
            
        return pd.DataFrame(jira_comments)


# ## Extract data

# In[7]:


# Connection to Presto
presto = PrestoConnection()


# In[ ]:


# Connection to Jira API
jira = JiraConnection()


# In[ ]:


"""
Load Meeseeks data
"""

# Define data range
start_date = "2021-04-01-00"
end_date = datetime.date.today().strftime("%Y-%m-%d-00")

# Load incident to incident group dataframe
incident_group_matches = presto.get_incident_group_matches(start_date, end_date)

# Load incident group to jira ticket dataframe
jira_group_matches = presto.get_jira_group_matches(start_date, end_date)

# Load incident groups metadata and comment information dataframe
incident_groups = presto.get_incident_groups(start_date, end_date)


# In[ ]:


"""
Load Jira-Meeseeks tickets data
"""

# If data previously saved, skip extraction
reload_data = False
if reload_data:
    # Feed jira_group_matches dataframe to load jira tickets associated to incident groups
    jira_meeseeks_df = jira.get_jira_text(jira_group_matches)
    
    # Save data
    jira_meeseeks_df.to_csv("./data/jira_meeseeks.csv")
    
# Load data
jira_meeseeks_df = pd.read_csv("./data/jira_meeseeks.csv").drop("Unnamed: 0", axis=1)
jira_meeseeks_df["jira_label"] = jira_meeseeks_df["jira_label"].mask(jira_meeseeks_df["jira_label"].isna(), "")


# In[ ]:


"""
Attach labels to Meeseeks data

Note: This is specific to the CSV file containing labeled incidents. Adjust accordingly.
"""

# Standardized label mapping
manual_root_cause_mapping = {"Noise" : "Noise",
                             "Noise - auto recover" : "Noise",
                             "Downstream" : "Downstream",
                             "Downstream - Different" : "Downstream",
                             "Downstream - different" : "Downstream",
                             "Downstream - Capacity" : "Downstream", 
                             "Downstream - Dyno" : "Downstream",
                             "Deploy" : "Deploy",
                             "Deploy - Canary" : "Deploy",
                             "Deploy - Config" : "Deploy",
                             "Deploy - restart" : "Deploy",
                             "Outlier Bad Hosts" : "Outlier Bad Hosts",
                             "Outlier Host" : "Outlier Bad Hosts",
                             "Outlier Host - GC" : "Outlier Bad Hosts",
                             "GC" : "Garbage Collection",
                             "Capacity" : "Capacity",
                             "Capacity - Quota" : "Capacity",
                             "Database" : "Database",
                             "Database - MySQL" : "Database",
                             "Upstream" : "Upstream",
                             "CM" : "CM",
                             "CM - Planned Maintenance" : "CM",
                             "CM - Traffic Shift" : "CM",
                             "CM - Load Test" : "CM",
                             "Couchbase" : "Database",
                             "Espresso" : "Database"}

# Load labeled incidents data
manual_label_dir = "./data/panel_title_data_2021-07-22.csv"
labels_df = pd.read_csv(manual_label_dir).drop("id", axis=1)[["incident_id","label"]]
labels_df["label"] = labels_df["label"].astype(str).apply(str.strip)

# Merge labeled incidents data with incident_group cross-reference table
labels_df = pd.merge(incident_group_matches[["incident_group_id", "incident_id"]], labels_df, how="right", on="incident_id")
labels_df = labels_df[labels_df["incident_group_id"].notna()]
labels_df.incident_group_id = labels_df.incident_group_id.astype('int64')

# Group incidents into incident group and extract first label in group
# This assumes all incidents in an incident group have the same label
labels_df = labels_df[["incident_group_id","label"]].groupby('incident_group_id').first().reset_index()
labels_df = labels_df[labels_df["label"].notna()]

# Map labels to standardized labels
labels_df["label"] = labels_df["label"].apply(lambda x: x.strip())
labels_df["label"] = labels_df["label"].apply(lambda x: x if x=="" else manual_root_cause_mapping.get(x.strip(), "Other"))

# Merge labeled incident groups to incident groups data
incidents_df = pd.merge(labels_df, incident_groups, how="right", on="incident_group_id")


# In[ ]:


"""
Add additional incident group labels

Note: This is specific to the labeled incident groups. Adjust accordingly.
"""

# Load manually labeled incident groups data
manual_label_dir2 = "./data/labeled_groups.csv"
labels2_df = pd.read_csv(manual_label_dir2).drop("Unnamed: 0", axis=1)

# Pre-process labels
labels2_df = labels2_df[~labels2_df["actual"].isin(["?","I don't know"])]
labels2_df = labels2_df[labels2_df["actual"].notna()]
labels2_df["actual"] = labels2_df["actual"].astype(str).apply(lambda x: x.split()[0])
labels2_df = labels2_df[["incident_group_id", "actual"]]

# Merge with labeled incident groups data
incidents_df = pd.merge(incidents_df, labels2_df, how="left", on="incident_group_id")

# Include additional labels in label column where labels did not exist
incidents_df["label"] = incidents_df["label"].mask((incidents_df["label"]=="") | (incidents_df["label"].isna()), incidents_df["actual"])
incidents_df["label"] = incidents_df["label"].mask(incidents_df["label"].isna(), "")
incidents_df = incidents_df.drop("actual", axis=1)


# In[ ]:


"""
Define final incident groups data format. 
"""

# Define text field
incidents_df["text"] = incidents_df["comment"]

# Choose final data fields
incidents_df = incidents_df[["incident_group_id", "label", "text"]]


# In[ ]:


"""
Merge incident groups with additional Jira-Meeseeks text data
"""

# Merge jira-meeseeks tickets data with jira_group_matches cross-reference table
jira_text_df = pd.merge(jira_group_matches[["incident_group_id","jira_ticket_id"]], jira_meeseeks_df, how="left", on="jira_ticket_id")

# Merge incident groups data with cross-referenced jira tickets data
incidents_df = pd.merge(incidents_df, jira_text_df, how="left", on="incident_group_id")

# Pre-process jira ticket text and append to text field
incidents_df["jira_text"] = incidents_df["jira_text"].mask(incidents_df["jira_text"].isna(), "").astype(str)
incidents_df["text"] = incidents_df[["text", "jira_text"]].agg(' '.join, axis=1)

# Pre-process labels
incidents_df["label"] = incidents_df["label"].mask((incidents_df["label"]=="") | (incidents_df["label"].isna()), incidents_df["jira_label"])
incidents_df["label"] = incidents_df["label"].mask(incidents_df["label"].isna(), "")
incidents_df["label"] = incidents_df["label"].apply(lambda x: x.strip())
incidents_df["label"] = incidents_df["label"].apply(lambda x: x if x=="" else manual_root_cause_mapping.get(x.strip(), "Other"))

# Choose final data fields
incidents_df = incidents_df[["incident_group_id", "label", "text"]]


# In[ ]:


# Save incident groups data
incidents_df.to_csv("./data/incidents.csv")


# ## Load final data

# In[8]:


"""
Text preprocessing methods
"""
from gensim.parsing.preprocessing import (strip_punctuation, strip_multiple_whitespaces,
                                          strip_numeric, remove_stopwords, strip_short, stem_text)

def strip_links(s):
    s = re.sub(r"http\S+", "", s)
    l = []
    for word in s.split():
        if '.com' not in word:
            l.append(word)
    return ' '.join(l)

def strip_control_characters(s):
    s_new = ""
    for ch in s:
        if unicodedata.category(ch)[0] == "C":
            s_new += " "
        else:
            s_new += ch
            
    s_new = s_new.replace(u'\xa0', u' ')
    return s_new

def preprocess_text(s):
    s = strip_control_characters(s)
    s = strip_links(s)
    s = strip_punctuation(s)
    s = strip_multiple_whitespaces(s)
    s = strip_numeric(s)
    s = strip_short(s, minsize=3)
    s = remove_stopwords(s)
    return s.strip().lower()


# In[9]:


"""
Load incidents data
"""

# Read pre-saved data incidents
incidents_df = pd.read_csv("./data/incidents.csv").drop("Unnamed: 0", axis=1)
incidents_df["label"] = incidents_df["label"].mask(incidents_df["label"].isna(), "")
incidents_df["text"] = incidents_df["text"].astype(str)

# Preprocess text
incidents_df["text"] = incidents_df["text"].apply(lambda s: preprocess_text(s))

# Only keep incident groups with non-empty comment
incidents_df = incidents_df[incidents_df["text"] != ""].copy()

# Restrict to a select list of labels. Cluster other labels into "Other" label
keep_labels = ["Noise", "Downstream", "Deploy", "Outlier Bad Hosts", ""]
incidents_df["label"] = incidents_df["label"].apply(lambda x: x if x in keep_labels else "Other")

# Define labels list and label index mappings
labels_list = ["Noise", "Downstream", "Deploy", "Outlier Bad Hosts", "Other"]
label_map = {label:idx for idx, label in enumerate(labels_list)}
inv_label_map = {v:k for k, v in label_map.items()}

# Create label index field. Define "-1" for incident groups without label
incidents_df["target"] = incidents_df["label"].apply(lambda x: label_map[x] if x in label_map else -1)

# Choose final data fields
incidents_df = incidents_df[["incident_group_id","label","target","text"]]


# In[10]:


"""
Load additional Jira comments data
"""

def convert_to_str(x):
    if not x:
        return ''
    try:
        return str(x)   
    except:        
        return ''
    
# Load data containing unlabeled jira tickets
jira_data_dir = "./data/jira.csv"
jira_df = pd.read_csv(jira_data_dir, converters={'summary':convert_to_str,'description':convert_to_str})

# Define jira data texts field
jira_df["text"] = jira_df[["summary", "description"]].agg(' '.join, axis=1)

# Pre-process text
jira_df["text"] = jira_df["text"].apply(lambda s: preprocess_text(s))

# Only keep tickets with non-empty text
jira_df = jira_df[jira_df["text"] != ""]

# Choose final data fields
jira_df = jira_df[["ticket_num","text"]]


# In[11]:


"""
Create text file with all available text.
To be used in pre-training for language model.
"""

# Define all text data available
all_text = incidents_df["text"].to_list() + jira_df["text"].to_list()

# Save data
save_all_text_dir = "./data/all_text.txt"
with open(save_all_text_dir, 'w') as f:
    for text in all_text:
        f.write("%s\n" % text)


# In[15]:


jira_data_dir = "./data/jira.csv"
jira_df = pd.read_csv(jira_data_dir, converters={'summary':convert_to_str,'description':convert_to_str})
jira_df["ticket_num"].to_csv("./data/jira_ticket_num.csv")


# # Language Model Pre-training

# ## Masked language modeling (MLM)

# In[ ]:


"""
This script runs additional MLM pre-training of a pre-trained
DistilBERT model using text data saved in ./data/all_text.txt.

Trained for 3 epochs.
Used batch size equal to 8.
Resulting model saved in ./data/tmp/meeseeks-mlm-nostop.
These parameters can be adjusted.

Training time takes around 2 hours on 8GB M60 GPU.
"""

get_ipython().system('python run_mlm.py     --model_name_or_path distilbert-base-uncased     --train_file ./data/all_text.txt     --do_train     --do_eval     --output_dir ./distilbert-meeseeks-mlm     --line_by_line     --overwrite_output_dir     --per_device_train_batch_size=8     --per_device_eval_batch_size=8     --num_train_epochs=3')


# # Training / Fine-tuning with Cross-validation

# In[ ]:


"""
Define labeled data
"""
dataset = incidents_df[incidents_df["label"]!=""]
texts = dataset["text"].to_list()
labels = dataset["label"].to_list()
targets = dataset["target"].to_list()


# # Supervised Text Classification (STC)

# In[ ]:


from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support


# In[ ]:


"""
Helper methods
"""

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # calculate accuracy and f1 score using sklearn functions
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="micro")
    return {
        'accuracy': acc,
        'f1 score': f1,
    }

def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    # perform inference to our model
    outputs = model(**inputs)
    
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    
    # executing argmax function to get the candidate label
    return inv_label_map[probs.argmax()]


# In[ ]:


"""
Extend Pytorch Dataset class for classification purpose
"""

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# ## DistilBERT with MLM pre-training + STC fine-tuning

# In[ ]:


"""
Fine-tune and evaluate DistilBERT model pre-trained on unlabeled data
"""

num_folds = 5    # number of folds for cross-validation
max_length = 512 # max sequence length for each document/sentence sample
batch_size = 8   # training and evaluation batch size
epochs = 32      # number of epochs to fine-tune
valid_size = 16  # validation data size, use for early stopping
early_stopping_patience = 5 # early stopping epoch patience before terminating
learning_rate = 1e-5 # learning rate
pretrained_model_dir = "./distilbert-meeseeks-mlm" # MLM pre-trained model directory

# store global results
y_true = [""]*len(texts)
y_pred = [""]*len(texts)

# store list of fold-averaged metrics
list_pre_fine_acc = []
list_pre_fine_prec_avg, list_pre_fine_rec_avg, list_pre_fine_f1_avg = [], [], []
list_pre_fine_prec, list_pre_fine_rec, list_pre_fine_f1 = [], [], []

# train and evaluate using cross validation (using stratified k-fold, can use regular k-fold)
kf = StratifiedKFold(n_splits=num_folds)
for train_index, test_index in tqdm(kf.split(texts, targets), total=kf.get_n_splits(texts, targets)):
    
    # define train and test data
    train_texts, test_texts = [texts[idx] for idx in train_index], [texts[idx] for idx in test_index]
    train_targets, test_targets = [targets[idx] for idx in train_index], [targets[idx] for idx in test_index]
    train_labels, test_labels = [labels[idx] for idx in train_index], [labels[idx] for idx in test_index]
    
    # define train and train-validation data
    valid_size = valid_size
    (train_texts, valid_texts, train_targets, valid_targets, train_labels, valid_labels) = train_test_split(train_texts, train_targets, train_labels, test_size=valid_size, random_state=1, stratify=train_targets)
    
    # load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_dir, num_labels=len(label_map))
    model.to("cuda")
    model.train()

    # tokenize train and validation data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

    # convert our tokenized data into a torch Dataset
    train_dataset = Dataset(train_encodings, train_targets)
    valid_dataset = Dataset(valid_encodings, valid_targets)

    # define training arguments
    training_args = TrainingArguments(
        report_to=None,
        output_dir='./results',                  # output directory
        overwrite_output_dir = True,             # overwrite output directory
        do_train=True,                           # train
        do_eval=True,                            # evaluate during training
        num_train_epochs=epochs,                 # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        warmup_steps=250,                        # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                       # strength of weight decay
        learning_rate=learning_rate,             # set learning rate 
        logging_dir='./logs',                    # directory for storing logs
        load_best_model_at_end=True,             # load the best model when finished training (default metric is loss)
        metric_for_best_model='eval_loss',       # validation metric for model selection
        save_strategy="epoch",                   # save based on epochs
        logging_strategy="epoch",                # log based on epochs
        evaluation_strategy="epoch",             # evaluate base on epochs
#         logging_steps=logging_steps,             # log & save weights each logging_steps

    )

    # define trainer
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        callbacks = [EarlyStoppingCallback(early_stopping_patience = early_stopping_patience)] # early stopping callback
    )

    # train and evaluate after training is done
    trainer.train()
    print(trainer.evaluate())

    # save the fine tuned model & tokenizer
    model_path = "./meeseeks-distilbert-base-uncased"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # load saved model & tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to("cuda")

    # make predictions on held out test data set
    data_loader = torch.utils.data.DataLoader(test_texts, batch_size=batch_size, shuffle=False)
    predictions = []
    for batch in tqdm(data_loader):
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)
        preds = [inv_label_map[idx] for idx in probs.argmax(axis=1).cpu().numpy()]
        predictions += preds
    
    # store local predictions
    fold_y_pred = []
    fold_y_true = []
    for i, idx in enumerate(test_index):
        y_pred[idx] = predictions[i]
        y_true[idx] = test_labels[i]
        fold_y_pred.append(predictions[i])
        fold_y_true.append(test_labels[i])
        
    # compute fold metrics
    fold_acc = accuracy_score(fold_y_true, fold_y_pred)
    fold_prec_avg, fold_rec_avg, fold_f1_avg, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average='macro')
    fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average=None, labels=labels_list)
    
    # store results on list of fold metrics
    list_pre_fine_acc.append(fold_acc)
    list_pre_fine_prec_avg.append(fold_prec_avg)
    list_pre_fine_rec_avg.append(fold_rec_avg)
    list_pre_fine_f1_avg.append(fold_f1_avg)
    list_pre_fine_prec.append(fold_prec)
    list_pre_fine_rec.append(fold_rec)
    list_pre_fine_f1.append(fold_f1)


# In[ ]:


# print global metrics
classification_report(y_true, y_pred)


# In[ ]:


# compute average and stdev of metrics over folds
pre_fine_acc_mean = np.mean(list_pre_fine_acc)
pre_fine_acc_std = np.std(list_pre_fine_acc)
pre_fine_prec_avg_mean = np.mean(list_pre_fine_prec_avg)
pre_fine_prec_avg_std = np.std(list_pre_fine_prec_avg)
pre_fine_rec_avg_mean = np.mean(list_pre_fine_rec_avg)
pre_fine_rec_avg_std = np.std(list_pre_fine_rec_avg)
pre_fine_f1_avg_mean = np.mean(list_pre_fine_f1_avg)
pre_fine_f1_avg_std = np.std(list_pre_fine_f1_avg)

pre_fine_prec_mean = np.mean(list_pre_fine_prec, axis=0)
pre_fine_prec_std = np.std(list_pre_fine_prec, axis=0)
pre_fine_rec_mean = np.mean(list_pre_fine_rec, axis=0)
pre_fine_rec_std = np.std(list_pre_fine_rec, axis=0)
pre_fine_f1_mean = np.mean(list_pre_fine_f1, axis=0)
pre_fine_f1_std = np.std(list_pre_fine_f1, axis=0)


# In[ ]:


# store results in pickle file
# done because it takes a while to compute results and notebook can crash
with open('./data/pre_fine.pkl', 'wb') as f:
    pickle.dump([pre_fine_acc, pre_fine_prec_avg, pre_fine_rec_avg, pre_fine_f1_avg, pre_fine_prec, pre_fine_rec, pre_fine_f1,                 list_pre_fine_acc, list_pre_fine_prec_avg, list_pre_fine_rec_avg, list_pre_fine_f1_avg, list_pre_fine_prec, list_pre_fine_rec, list_pre_fine_f1], f)


# In[ ]:


# load results saved in pickle file
with open('./data/pre_fine.pkl', 'rb') as f:
    pre_fine_acc, pre_fine_prec_avg, pre_fine_rec_avg, pre_fine_f1_avg, pre_fine_prec, pre_fine_rec, pre_fine_f1,                 list_pre_fine_acc, list_pre_fine_prec_avg, list_pre_fine_rec_avg, list_pre_fine_f1_avg, list_pre_fine_prec, list_pre_fine_rec, list_pre_fine_f1 = pickle.load(f)


# In[ ]:


# clear GPU memory
del trainer, model, tokenizer
del train_dataset, valid_dataset
del train_encodings, valid_encodings
del data_loader
del inputs, outputs, probs
torch.cuda.empty_cache()


# ## DistilBERT with STC fine-tuning

# In[ ]:


"""
Fine-tune and evaluate DistilBERT model pre-trained on unlabeled data
"""

num_folds = 5    # number of folds for cross-validation
max_length = 512 # max sequence length for each document/sentence sample
batch_size = 8   # training and evaluation batch size
epochs = 32      # number of epochs to fine-tune
valid_size = 16  # validation data size, use for early stopping
early_stopping_patience = 5 # early stopping epoch patience before terminating
learning_rate = 1e-5 # learning rate

# store global results
y_true = [""]*len(texts)
y_pred = [""]*len(texts)

# store list of fold-averaged metrics
list_fine_acc = []
list_fine_prec_avg, list_fine_rec_avg, list_fine_f1_avg = [], [], []
list_fine_prec, list_fine_rec, list_fine_f1 = [], [], []

# train and evaluate using cross validation (using stratified k-fold, can use regular k-fold)
kf = StratifiedKFold(n_splits=num_folds)
for train_index, test_index in tqdm(kf.split(texts, targets), total=kf.get_n_splits(texts, targets)):
    
    # define train and test data
    train_texts, test_texts = [texts[idx] for idx in train_index], [texts[idx] for idx in test_index]
    train_targets, test_targets = [targets[idx] for idx in train_index], [targets[idx] for idx in test_index]
    train_labels, test_labels = [labels[idx] for idx in train_index], [labels[idx] for idx in test_index]
    
    # define train and train-validation data
    valid_size = valid_size
    (train_texts, valid_texts, train_targets, valid_targets, train_labels, valid_labels) = train_test_split(train_texts, train_targets, train_labels, test_size=valid_size, random_state=1, stratify=train_targets)
    
    # load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_map))
    model.to("cuda")
    model.train()

    # tokenize train and validation data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

    # convert our tokenized data into a torch Dataset
    train_dataset = Dataset(train_encodings, train_targets)
    valid_dataset = Dataset(valid_encodings, valid_targets)

    # define training arguments
    training_args = TrainingArguments(
        report_to=None,
        output_dir='./results',                  # output directory
        overwrite_output_dir = True,             # overwrite output directory
        do_train=True,                           # train
        do_eval=True,                            # evaluate during training
        num_train_epochs=epochs,                 # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        warmup_steps=250,                        # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                       # strength of weight decay
        learning_rate=learning_rate,             # set learning rate 
        logging_dir='./logs',                    # directory for storing logs
        load_best_model_at_end=True,             # load the best model when finished training (default metric is loss)
        metric_for_best_model='eval_loss',       # validation metric for model selection
        save_strategy="epoch",                   # save based on epochs
        logging_strategy="epoch",                # log based on epochs
        evaluation_strategy="epoch",             # evaluate base on epochs
#         logging_steps=logging_steps,             # log & save weights each logging_steps

    )

    # define trainer
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
        callbacks = [EarlyStoppingCallback(early_stopping_patience = early_stopping_patience)] # early stopping callback
    )

    # train and evaluate after training is done
    trainer.train()
    print(trainer.evaluate())

    # save the fine tuned model & tokenizer
    model_path = "./meeseeks-distilbert-base-uncased-nomlm"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # load saved model & tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to("cuda")

    # make predictions on held out test data set
    data_loader = torch.utils.data.DataLoader(test_texts, batch_size=batch_size, shuffle=False)
    predictions = []
    for batch in tqdm(data_loader):
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)
        preds = [inv_label_map[idx] for idx in probs.argmax(axis=1).cpu().numpy()]
        predictions += preds
    
    # store local predictions
    fold_y_pred = []
    fold_y_true = []
    for i, idx in enumerate(test_index):
        y_pred[idx] = predictions[i]
        y_true[idx] = test_labels[i]
        fold_y_pred.append(predictions[i])
        fold_y_true.append(test_labels[i])
        
    # compute fold metrics
    fold_acc = accuracy_score(fold_y_true, fold_y_pred)
    fold_prec_avg, fold_rec_avg, fold_f1_avg, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average='macro')
    fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average=None, labels=labels_list)
    
    # store results on list of fold metrics
    list_fine_acc.append(fold_acc)
    list_fine_prec_avg.append(fold_prec_avg)
    list_fine_rec_avg.append(fold_rec_avg)
    list_fine_f1_avg.append(fold_f1_avg)
    list_fine_prec.append(fold_prec)
    list_fine_rec.append(fold_rec)
    list_fine_f1.append(fold_f1)


# In[ ]:


# print global metrics
print(classification_report(y_true, y_pred))


# In[ ]:


# compute average and stdev of metrics over folds
fine_acc_mean = np.mean(list_fine_acc)
fine_acc_std = np.std(list_fine_acc)
fine_prec_avg_mean = np.mean(list_fine_prec_avg)
fine_prec_avg_std = np.std(list_fine_prec_avg)
fine_rec_avg_mean = np.mean(list_fine_rec_avg)
fine_rec_avg_std = np.std(list_fine_rec_avg)
fine_f1_avg_mean = np.mean(list_fine_f1_avg)
fine_f1_avg_std = np.std(list_fine_f1_avg)

fine_prec_mean = np.mean(list_fine_prec, axis=0)
fine_prec_std = np.std(list_fine_prec, axis=0)
fine_rec_mean = np.mean(list_fine_rec, axis=0)
fine_rec_std = np.std(list_fine_rec, axis=0)
fine_f1_mean = np.mean(list_fine_f1, axis=0)
fine_f1_std = np.std(list_fine_f1, axis=0)


# In[ ]:


# store results in pickle file
# done because it takes a while to compute results and notebook can crash
with open('./data/fine.pkl', 'wb') as f:
    pickle.dump([fine_acc, fine_prec_avg, fine_rec_avg, fine_f1_avg, fine_prec, fine_rec, fine_f1,                 list_fine_acc, list_fine_prec_avg, list_fine_rec_avg, list_fine_f1_avg, list_fine_prec, list_fine_rec, list_fine_f1], f)


# In[ ]:


# load results saved in pickle file
with open('./data/fine.pkl', 'rb') as f:
    fine_acc, fine_prec_avg, fine_rec_avg, fine_f1_avg, fine_prec, fine_rec, fine_f1,                 list_fine_acc, list_fine_prec_avg, list_fine_rec_avg, list_fine_f1_avg, list_fine_prec, list_fine_rec, list_fine_f1 = pickle.load(f)


# In[ ]:


# clear GPU memory
del trainer, model, tokenizer
del train_dataset, valid_dataset
del train_encodings, valid_encodings
del data_loader
del inputs, outputs, probs
torch.cuda.empty_cache()


# ## TF-IDF classifier

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[ ]:


"""
Train logistic classifier with TF-IDF features
"""
num_folds = 5    # number of folds for cross-validation

# store global results
y_true = [""]*len(texts)
y_pred = [""]*len(texts)

# store list of fold-averaged metrics
list_tfidf_acc = []
list_tfidf_prec_avg, list_tfidf_rec_avg, list_tfidf_f1_avg = [], [], []
list_tfidf_prec, list_tfidf_rec, list_tfidf_f1 = [], [], []

# train and evaluate using cross validation (using stratified k-fold, can use regular k-fold)
kf = StratifiedKFold(n_splits=num_folds)
for train_index, test_index in tqdm(kf.split(texts, targets), total=kf.get_n_splits(texts, targets)):
    
    # define train and test data
    train_texts, test_texts = [texts[idx] for idx in train_index], [texts[idx] for idx in test_index]
    train_targets, test_targets = [targets[idx] for idx in train_index], [targets[idx] for idx in test_index]
    train_labels, test_labels = [labels[idx] for idx in train_index], [labels[idx] for idx in test_index]
    
    # compute TF-IDF features based on training data
    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(train_texts)
    test_X = vectorizer.transform(test_texts)
    
    # fit logistic regression model and make predictions on test data
    logreg = LogisticRegression()
    logreg.fit(train_X, train_targets)
    predictions = logreg.predict(test_X)
        
    # store local predictions
    fold_y_pred = []
    fold_y_true = []
    for i, idx in enumerate(test_index):
        y_pred[idx] = inv_label_map[predictions[i]]
        y_true[idx] = test_labels[i]
        fold_y_pred.append(inv_label_map[predictions[i]])
        fold_y_true.append(test_labels[i])
        
    # compute fold metrics
    fold_acc = accuracy_score(fold_y_true, fold_y_pred)
    fold_prec_avg, fold_rec_avg, fold_f1_avg, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average='macro')
    fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average=None, labels=labels_list)
    
    # store results on list of fold metrics
    list_tfidf_acc.append(fold_acc)
    list_tfidf_prec_avg.append(fold_prec_avg)
    list_tfidf_rec_avg.append(fold_rec_avg)
    list_tfidf_f1_avg.append(fold_f1_avg)
    list_tfidf_prec.append(fold_prec)
    list_tfidf_rec.append(fold_rec)
    list_tfidf_f1.append(fold_f1)


# In[ ]:


# print global metrics
classification_report(y_true, y_pred)


# In[ ]:


# compute average and stdev of metrics over folds
tfidf_acc_mean = np.mean(list_tfidf_acc)
tfidf_acc_std = np.std(list_tfidf_acc)
tfidf_prec_avg_mean = np.mean(list_tfidf_prec_avg)
tfidf_prec_avg_std = np.std(list_tfidf_prec_avg)
tfidf_rec_avg_mean = np.mean(list_tfidf_rec_avg)
tfidf_rec_avg_std = np.std(list_tfidf_rec_avg)
tfidf_f1_avg_mean = np.mean(list_tfidf_f1_avg)
tfidf_f1_avg_std = np.std(list_tfidf_f1_avg)

tfidf_prec_mean = np.mean(list_tfidf_prec, axis=0)
tfidf_prec_std = np.std(list_tfidf_prec, axis=0)
tfidf_rec_mean = np.mean(list_tfidf_rec, axis=0)
tfidf_rec_std = np.std(list_tfidf_rec, axis=0)
tfidf_f1_mean = np.mean(list_tfidf_f1, axis=0)
tfidf_f1_std = np.std(list_tfidf_f1, axis=0)


# ## Doc2vec classifier

# In[ ]:


import nltk
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.linear_model import LogisticRegression


# In[ ]:


"""
Train a logistic classifier with doc2vec embeddings
"""
num_folds = 5    # number of folds for cross-validation
epochs = 32      # number of epochs to train doc2vec model

# Method to tokenize text
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

# store global results
y_true = [""]*len(texts)
y_pred = [""]*len(texts)

# store list of fold-averaged metrics
list_doc2vec_acc = []
list_doc2vec_prec_avg, list_doc2vec_rec_avg, list_doc2vec_f1_avg = [], [], []
list_doc2vec_prec, list_doc2vec_rec, list_doc2vec_f1 = [], [], []

# train and evaluate using cross validation (using stratified k-fold, can use regular k-fold)
kf = StratifiedKFold(n_splits=num_folds)
for train_index, test_index in tqdm(kf.split(texts, targets), total=kf.get_n_splits(texts, targets)):
    
    # define train and test data
    train_texts, test_texts = [texts[idx] for idx in train_index], [texts[idx] for idx in test_index]
    train_targets, test_targets = [targets[idx] for idx in train_index], [targets[idx] for idx in test_index]
    train_labels, test_labels = [labels[idx] for idx in train_index], [labels[idx] for idx in test_index]
    
    # create tagged documents dataframes
    train_df = pd.DataFrame(list(zip(train_labels, train_targets, train_texts)), columns =['label', 'target', 'text'])
    test_df = pd.DataFrame(list(zip(test_labels, test_targets, test_texts)), columns =['label', 'target', 'text'])
    train_tagged = train_df.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r['label']]), axis=1)
    test_tagged = test_df.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r['label']]), axis=1)

    # train doc2vec vectorizer on training data
    model_dbow = Doc2Vec(workers=multiprocessing.cpu_count())
    model_dbow.build_vocab(train_tagged.values)
    model_dbow.train(train_tagged.values, total_examples=len(train_tagged.values), epochs=epochs)
    
    # compute doc2vec features
    y_train, X_train = zip(*[(doc.tags[0], model_dbow.infer_vector(doc.words, steps=20)) for doc in train_tagged.values])
    y_test, X_test = zip(*[(doc.tags[0], model_dbow.infer_vector(doc.words, steps=20)) for doc in test_tagged.values])

    # fit logistic regression model and make predictions on test data
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    
    # store local predictions      
    fold_y_pred = []
    fold_y_true = []
    for i, idx in enumerate(test_index):
        y_pred[idx] = predictions[i]
        y_true[idx] = test_labels[i]
        fold_y_pred.append(predictions[i])
        fold_y_true.append(test_labels[i])
      
    # compute fold metrics
    fold_acc = accuracy_score(fold_y_true, fold_y_pred)
    fold_prec_avg, fold_rec_avg, fold_f1_avg, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average='macro')
    fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average=None, labels=labels_list)
    
    # store results on list of fold metrics
    list_doc2vec_acc.append(fold_acc)
    list_doc2vec_prec_avg.append(fold_prec_avg)
    list_doc2vec_rec_avg.append(fold_rec_avg)
    list_doc2vec_f1_avg.append(fold_f1_avg)
    list_doc2vec_prec.append(fold_prec)
    list_doc2vec_rec.append(fold_rec)
    list_doc2vec_f1.append(fold_f1)


# In[ ]:


# print global metrics
classification_report(y_true, y_pred)


# In[ ]:


# compute average and stdev of metrics over folds
doc2vec_acc_mean = np.mean(list_doc2vec_acc)
doc2vec_acc_std = np.std(list_doc2vec_acc)
doc2vec_prec_avg_mean = np.mean(list_doc2vec_prec_avg)
doc2vec_prec_avg_std = np.std(list_doc2vec_prec_avg)
doc2vec_rec_avg_mean = np.mean(list_doc2vec_rec_avg)
doc2vec_rec_avg_std = np.std(list_doc2vec_rec_avg)
doc2vec_f1_avg_mean = np.mean(list_doc2vec_f1_avg)
doc2vec_f1_avg_std = np.std(list_doc2vec_f1_avg)

doc2vec_prec_mean = np.mean(list_doc2vec_prec, axis=0)
doc2vec_prec_std = np.std(list_doc2vec_prec, axis=0)
doc2vec_rec_mean = np.mean(list_doc2vec_rec, axis=0)
doc2vec_rec_std = np.std(list_doc2vec_rec, axis=0)
doc2vec_f1_mean = np.mean(list_doc2vec_f1, axis=0)
doc2vec_f1_std = np.std(list_doc2vec_f1, axis=0)


# ## Stratified dummy classifier

# In[ ]:


from sklearn.dummy import DummyClassifier


# In[ ]:


"""
Run a dummy classifier
"""

# store global results
y_true = [""]*len(texts)
y_pred = [""]*len(texts)

# store list of fold-averaged metrics
list_dummy_acc = []
list_dummy_prec_avg, list_dummy_rec_avg, list_dummy_f1_avg = [], [], []
list_dummy_prec, list_dummy_rec, list_dummy_f1 = [], [], []

# train and evaluate using cross validation (using stratified k-fold, can use regular k-fold)
kf = StratifiedKFold(n_splits=num_folds)
for train_index, test_index in tqdm(kf.split(texts, targets), total=kf.get_n_splits(texts, targets)):
    
    # define train and test data
    train_texts, test_texts = [texts[idx] for idx in train_index], [texts[idx] for idx in test_index]
    train_targets, test_targets = [targets[idx] for idx in train_index], [targets[idx] for idx in test_index]
    train_labels, test_labels = [labels[idx] for idx in train_index], [labels[idx] for idx in test_index]
    
    # fit dummy classifier and make predictiosn on test data
    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit([0]*len(train_texts), train_targets)
    predictions = dummy_clf.predict([0]*len(test_texts))
    
    # store local predictions
    fold_y_pred = []
    fold_y_true = []
    for i, idx in enumerate(test_index):
        y_pred[idx] = inv_label_map[predictions[i]]
        y_true[idx] = test_labels[i]
        fold_y_pred.append(inv_label_map[predictions[i]])
        fold_y_true.append(test_labels[i])
        
    # compute fold metrics
    fold_acc = accuracy_score(fold_y_true, fold_y_pred)
    fold_prec_avg, fold_rec_avg, fold_f1_avg, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average='macro')
    fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(fold_y_true, fold_y_pred, average=None, labels=labels_list)
    
    # store results on list of fold metrics
    list_dummy_acc.append(fold_acc)
    list_dummy_prec_avg.append(fold_prec_avg)
    list_dummy_rec_avg.append(fold_rec_avg)
    list_dummy_f1_avg.append(fold_f1_avg)
    list_dummy_prec.append(fold_prec)
    list_dummy_rec.append(fold_rec)
    list_dummy_f1.append(fold_f1)


# In[ ]:


# print global metrics
classification_report(y_true, y_pred)


# In[ ]:


# compute average and stdev of metrics over folds
dummy_acc_mean = np.mean(list_dummy_acc)
dummy_acc_std = np.std(list_dummy_acc)
dummy_prec_avg_mean = np.mean(list_dummy_prec_avg)
dummy_prec_avg_std = np.std(list_dummy_prec_avg)
dummy_rec_avg_mean = np.mean(list_dummy_rec_avg)
dummy_rec_avg_std = np.std(list_dummy_rec_avg)
dummy_f1_avg_mean = np.mean(list_dummy_f1_avg)
dummy_f1_avg_std = np.std(list_dummy_f1_avg)

dummy_prec_mean = np.mean(list_dummy_prec, axis=0)
dummy_prec_std = np.std(list_dummy_prec, axis=0)
dummy_rec_mean = np.mean(list_dummy_rec, axis=0)
dummy_rec_std = np.std(list_dummy_rec, axis=0)
dummy_f1_mean = np.mean(list_dummy_f1, axis=0)
dummy_f1_std = np.std(list_dummy_f1, axis=0)


# # Plot results

# In[ ]:


# create dataframe of metric means
df = pd.DataFrame({
    'Score': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Dummy stratified': [dummy_acc_mean, dummy_prec_avg_mean, dummy_rec_avg_mean, dummy_f1_avg_mean],
    'Doc2vec': [doc2vec_acc_mean, doc2vec_prec_avg_mean, doc2vec_rec_avg_mean, doc2vec_f1_avg_mean],
    'TF-IDF': [tfidf_acc_mean, tfidf_prec_avg_mean, tfidf_rec_avg_mean, tfidf_f1_avg_mean],
    'DistilBERT STC ': [fine_acc_mean, fine_prec_avg_mean, fine_rec_avg_mean, fine_f1_avg_mean],
    'DistilBERT MLM+STC': [pre_fine_acc_mean, pre_fine_prec_avg_mean, pre_fine_rec_avg_mean, pre_fine_f1_avg_mean],
})

# create dataframe of metric stdevs
df_std = pd.DataFrame({
    'Score': ['Accuracy', 'Precision', 'Recall', 'F1'],
    'Dummy stratified': [dummy_acc_std, dummy_prec_avg_std, dummy_rec_avg_std, dummy_f1_avg_std],
    'Doc2vec': [doc2vec_acc_std, doc2vec_prec_avg_std, doc2vec_rec_avg_std, doc2vec_f1_avg_std],
    'TF-IDF': [tfidf_acc_std, tfidf_prec_avg_std, tfidf_rec_avg_std, tfidf_f1_avg_std],
    'DistilBERT STC ': [fine_acc_std, fine_prec_avg_std, fine_rec_avg_std, fine_f1_avg_std],
    'DistilBERT MLM+STC': [pre_fine_acc_std, pre_fine_prec_avg_std, pre_fine_rec_avg_std, pre_fine_f1_avg_std],
})

# reformat dataframes
tidy = df.melt(id_vars='Score', var_name='Model').rename(columns=str.title)
tidy_std = df_std.melt(id_vars='Score', var_name='Model').rename(columns=str.title)

# create plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x='Score', y='Value', hue='Model', data=tidy, ax=ax)

# add error bars
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=tidy_std["Value"], fmt="none", c= "k", capsize=7, elinewidth=1.4)

# set plot limits
ax.set(ylim=(0,1))
plt.yticks(np.arange(0, 1.1, .1))
# plt.title('Scores')
sns.despine(fig)


# In[ ]:


# create dataframe of metric means
df = pd.DataFrame({
    'Label': labels_list,
    'Dummy stratified': dummy_f1_mean,
    'Doc2vec': doc2vec_f1_mean,
    'TF-IDF': tfidf_f1_mean,
    'DistilBERT STC': fine_f1_mean,
    'DistilBERT MLM+STC': pre_fine_f1_mean 
})

# create dataframe of metric stdevs
df_std = pd.DataFrame({
    'Label': labels_list,
    'Dummy stratified': dummy_f1_std,
    'Doc2vec': doc2vec_f1_std,
    'TF-IDF': tfidf_f1_std,
    'DistilBERT STC': fine_f1_std,
    'DistilBERT MLM+STC': pre_fine_f1_std 
})

# reformat dataframes
tidy = df.melt(id_vars='Label', value_name='F1', var_name='Model').rename(columns=str.title)
tidy_std = df_std.melt(id_vars='Label', value_name='F1', var_name='Model').rename(columns=str.title)

# order data labels according to data size
# Note: This is ad-hoc. Change according to chosen labels and their size.
ord_map = {"Deploy":0, "Other":1, "Downstream":2, "Noise":3, "Outlier Bad Hosts":4}
model_map = {"Dummy stratified":0, "Doc2vec":1, "TF-IDF":2, "DistilBERT STC":3, "DistilBERT MLM+STC":4}
tidy["target"] = tidy["Label"].apply(lambda x: ord_map[x])
tidy["model_id"] = tidy["Model"].apply(lambda x: model_map[x])
tidy = tidy.sort_values(by=["model_id", "target"])
tidy_std["target"] = tidy_std["Label"].apply(lambda x: ord_map[x])
tidy_std["model_id"] = tidy_std["Model"].apply(lambda x: model_map[x])
tidy_std = tidy_std.sort_values(by=["model_id", "target"])

# create plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(18, 9))
sns.barplot(x='Label', y='F1', hue='Model', data=tidy, ax=ax)

# add error bars
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=tidy_std["F1"], fmt="none", c= "k", capsize=7, elinewidth=1.4)

# set plot limits
ax.set(ylim=(0,1))
plt.yticks(np.arange(0, 1.1, .1))
# plt.title("F1 Score across labels")
sns.despine(fig)


# # Topic Modeling

# ## BERTopic

# In[ ]:


import umap
from bertopic import BERTopic
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from sentence_transformers import SentenceTransformer


# In[ ]:


# Define data
docs = incidents_df["text"].to_list()
labels = incidents_df["label"].to_list()
targets = incidents_df["target"].to_list()
classes = [label if label in label_map else "Unlabeled" for label in labels]


# ### With DistilBERT base

# In[ ]:


num_topics = len(labels_list)

# embedding model
model = TransformerDocumentEmbeddings("distilbert-base-uncased", layers="-1")

# fit and reduce topic model
topic_model = BERTopic(verbose=True, embedding_model=model)
topias, probs = topic_model.fit_transform(docs)
topics, probs = topic_model.reduce_topics(docs, topics, probs, nr_topics=num_topics)


# In[ ]:


# create embeddings
embeddings = []
for i in tqdm(range(len(docs))):
    sent = Sentence(docs[i])
    model.embed(sent)
    embeddings.append(sent.embedding.detach().cpu().numpy())
embeddings = np.array(embeddings)


# ### With DistilBERT finetuned

# In[ ]:


from transformers import DistilBertTokenizerFast, DistilBertModel


# In[ ]:


# load pretrained tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('./meeseeks-distilbert-base-uncased/')
model = DistilBertModel.from_pretrained('./meeseeks-distilbert-base-uncased').to("cuda")


# In[ ]:


# method to aggregate token embeddings across document
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# In[ ]:


# create embeddings
max_length = 512
embeddings = []
for i in tqdm(range(len(docs))):
    doc = docs[i]
    inputs = tokenizer(doc, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    outs = model(**inputs)
    embedding = mean_pooling(outs, inputs["attention_mask"])
    embedding = embedding.squeeze().detach().cpu().numpy()
    torch.cuda.synchronize()
    embeddings.append(embedding)
embeddings = np.array(embeddings)


# In[ ]:


# fit and reduce topic model
num_topics = len(labels_list)
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs, embeddings)
topics, probs = topic_model.reduce_topics(docs, topics, probs, nr_topics=num_topics)


# ## Analyze topics

# In[ ]:


topic_model.get_topic_info()


# In[ ]:


topic_model.visualize_topics()


# In[ ]:


for i in range(num_topics):
    print(i, topic_model.get_representative_docs(i))
    print()


# In[ ]:


for i in range(num_topics):
    print(i, [word for (word, prob) in topic_model.get_topic(i)[:10]])
    print()


# In[ ]:


topic_model.visualize_heatmap(n_clusters=num_topics)


# In[ ]:


mask = [i for i, cls in enumerate(classes) if cls!="Unlabeled"]
ndocs = [docs[i] for i in mask]
ntopics = [topics[i] for i in mask]
nclasses = [classes[i] for i in mask]
topics_per_class = topic_model.topics_per_class(ndocs, ntopics, classes=nclasses)
fig_semi_supervised = topic_model.visualize_topics_per_class(topics_per_class)
fig_semi_supervised


# ## Visualize topics in UMAP embedding

# In[ ]:


umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])


# In[ ]:


result["labels"] = topics
result["targets"] = labels
mapping = {'':'.',
           'Downstream':'v',
           'Noise':'x',
           'Deploy':'s',
           'Outlier Bad Hosts':'p',
           'Other':'+',
           'Garbage Collection':'v',
           'Capacity':'+',
           'Database':'s',
           'Upstream':'^',
           'CM':'p',
           'Infrastructure':'H'}


# In[ ]:


legend_labels = []
sns.reset_orig()
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
groups = clustered.groupby("targets")
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=5)
legend_labels.append("Outliers")
cmap = plt.get_cmap('rainbow', num_topics)
for (name, group) in groups:
    if name == "":
        plt.scatter(group.x, group.y, c=group.labels, s=15, cmap=cmap, marker=".")
        legend_labels.append("Unlabeled")
    else:
        plt.scatter(group.x, group.y, c=group.labels, s=130, cmap=cmap, marker=mapping[name])
        legend_labels.append(name)
ax.legend(labels=legend_labels)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0, cmap=cmap)
plt.xlim([-1,17])
plt.ylim([-3,18])
# fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: x)
plt.colorbar(ticks=np.arange(11))

