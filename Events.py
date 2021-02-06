#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df=pd.read_csv('events.csv')


# In[2]:


# On va convertir le timestamp pour obtenir une date plus précise

from datetime import datetime

df['date']=df['timestamp'].apply(lambda x : datetime.fromtimestamp(x/1000))

df.date

# Grâce aux attributs de datetime on peut isoler année, mois, jour et heure

df['year']=pd.to_datetime(df.date).dt.year
df['month']=pd.to_datetime(df.date).dt.month
df['day']=pd.to_datetime(df.date).dt.day
df['hour']=pd.to_datetime(df.date).dt.hour
df['weekday']=pd.to_datetime(df.date).dt.weekday
df['minutes']=pd.to_datetime(df.date).dt.minute
df['secondes']=pd.to_datetime(df.date).dt.second


# In[3]:


# On va avoir une date avec jour-mois-année

df['date']=df[['day','month','year']].sort_values(by=['day','month','year']).astype(str).agg('-'.join, axis=1)


# In[24]:


# Somme des doublons 

df.duplicated().sum()


# In[25]:


df.info()


# In[30]:


# Afficher la colonne transactionid,beaucoup de nan ne pas supprimer car on perd de l'infos

df['transactionid']


# In[31]:


#Nombre d'items qui apparaissent dans le dataset

df.itemid.value_counts()


# In[32]:


#Proportion d'event 

sns.countplot(x='event', data=df);
df.event.value_counts()


# In[33]:


#Proportion d'event en % graph circuliaire

plt.figure(figsize=(7,7))

plt.pie(df.event.value_counts(),
        labels=['view', 'addtocart','transaction'],
        colors=['steelblue','orange','green'],
        explode = [0.1, 0, 0],
        autopct=lambda x : str(round(x, 2)) + '%',
        pctdistance=0.7, labeldistance=1.3,
        shadow=True)
plt.legend();


# In[39]:


# Le nombre de transactions par mois

transac=df.loc[df['event']=='transaction']

sns.countplot(x='month', data=transac);


# In[40]:


fig=plt.figure(figsize=(18,10))

ax1 = fig.add_subplot(121) 
ax1=sns.countplot(x='day', data=transac) 
plt.title('Nombre de transactions par jours');

ax2 = fig.add_subplot(122) 
ax2=sns.countplot(x='hour', data=transac) 
plt.title('Nombre de transactions par heures');


# In[41]:


plt.figure(figsize=(16,8))
sns.countplot(x='day', data=transac)
plt.title('Nombre de transactions par jours');


# In[42]:


plt.figure(figsize=(16,8))
sns.countplot(x='weekday', data=transac)
plt.xticks(np.arange(7),['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
plt.title('Nombre de transactions par jours de la semaine');

# On remarque que les transactions sont plus en fin de semaine et début de semaine (Dimanche)


# In[43]:


plt.figure(figsize=(16,8))
sns.countplot(x='hour', data=transac);
plt.title('Nombre de transactions par heures');
# On remarque que les transactions se font plus entre 18h et 23h 


# In[44]:


# On va faire un histogramme et regrouper les heures 

sns.displot(transac.hour, bins=24, kde=True, rug=True, color='red')
plt.xticks([0, 6, 12, 18, 24],['Minuit', '6h', '12h', '18h', 'Minuit']);


# In[45]:


#Valeurs uniques

visitor=df.visitorid.unique()
print(visitor.size)

transaction=df.transactionid.unique()
print(transaction.size)

itemid=df.itemid.unique()
print(itemid.size)


# In[46]:


#Quel est le nb moyen de transaction par mois et par articles ?

item_transac_month=df.groupby(by=['month','itemid']).agg({'transactionid':'count'}).mean(level='month')

print(item_transac_month)


# In[ ]:





# In[4]:


# on récupère d'abord les view ensuite on va regarder le nb de view par articles
# nombre de vu par articles ?

view=df.loc[df['event']=='view']

nb_view=view.groupby(by=['itemid'], as_index=False).agg({'event':'count'}).rename(columns={'event':'nb_view'})

nb_view.max()


# In[ ]:





# In[5]:


# on récupère d'abord les addtocart ensuite on va regarder le nb de addtocart par articles
# nombre de parniers par articles ?

addtocartevent=df.loc[df['event']=='addtocart']

nb_add=addtocartevent.groupby(by=['itemid'], as_index=False).agg({'event':'count'}).rename(columns={'event':'nb_addtocart'})

nb_add.max()


# In[ ]:





# In[6]:


# on récupère d'abord les transactions ensuite on va regarder le nb de transactions par articles
# nombre de transactions par articles ?

transaction=df.loc[df['event']=='transaction']

nb_transac=transaction.groupby(by=['itemid'], as_index=False).agg({'event':'count'}).rename(columns={'event':'nb_transaction'})

nb_transac.sort_values('nb_transaction', ascending=False)

nb_transac.max()


# In[7]:


# On merge toute les bases pour avoir un suivi des objets selon leur statut

item_id=df.groupby(by='itemid', as_index=False).size()
item_id

item_event=pd.merge(item_id, nb_view, how='left')

item_event=pd.merge(item_event, nb_add, how='left')

item_event=pd.merge(item_event, nb_transac, how='left')

item_event.sort_values('nb_transaction', ascending=False)


# In[ ]:





# In[128]:


item_event.describe()


# In[129]:


#Quels sont les items les plus vus, ajoutés et transactions ? avec le max ? 

view_mode=item_event.loc[item_event['nb_view']==item_event['nb_view'].max()]
print(view_mode)

add_mode=item_event.loc[item_event['nb_addtocart']==item_event['nb_addtocart'].max()]
print(add_mode)

transac_mode=item_event.loc[item_event['nb_transaction']==item_event['nb_transaction'].max()]
print(transac_mode)


# In[4]:


# Nombre de views par visitorid, par item 

visit_view = df.loc[df['event'] == 'view'].groupby(by=['visitorid','itemid'], as_index=False).agg({'event':'count'})

visit_view = visit_view.rename(columns={"event": "nb_view"})

visit_view.sort_values('nb_view', ascending=False)


# In[5]:


# Nombre de paniers par visitorid, par item 

visit_add = df.loc[df['event'] == 'addtocart'].groupby(by=['visitorid','itemid'], as_index=False).agg({'event':'count'})

visit_add = visit_add.rename(columns={"event": "nb_addtocart"})

visit_add.sort_values('nb_addtocart', ascending=False)


# In[13]:


# Nombre de transacs par visitorid, par item 

visit_transac = df.loc[df['event'] == 'transaction'].groupby(by=['visitorid','itemid'], as_index=False).agg({'event':'count'})

visit_transac = visit_transac.rename(columns={"event": "nb_transac"})

visit_transac.sort_values('nb_transac', ascending=False)


# In[ ]:





# In[14]:



visit_item=pd.merge(visit_view, visit_add, how='outer')

visit_item=pd.merge(visit_item, visit_transac, how='outer')

visit_item.sort_values(['nb_transac','nb_addtocart', 'nb_view','visitorid', 'itemid'], ascending=[False, False, False, False, False])

visit_item.sort_values(by='nb_transac', ascending=False)

# Visiteur qui a acheté sans panier == peut être un acheteur compulsif ? 


# In[62]:


# On va faire un groupement des visitors qui achètent le plus 

visitor_transaction=visit_item.groupby(by='visitorid').agg({'nb_transac':'count', 'visitorid':'count'}).rename(columns={'visitorid':'nb_de_visit'})

plt.figure(figsize=(16,8))
plt.scatter(x='nb_de_visit', y='nb_transac', data=visitor_transaction)
plt.xlabel('Pour chaque visiteur unique, le nb de visites effectués')
plt.ylabel('Le nombre de transactions')
plt.title('Nombre visites par transactions');

vérif=visitor_transaction.loc[(visitor_transaction['nb_transac']==visitor_transaction['nb_transac'].max())]
print(vérif)


# In[60]:


# Test de Pearson ? 

from scipy.stats import pearsonr

pd.DataFrame(pearsonr(visitor_transaction['nb_de_visit'],visitor_transaction['nb_transac']), index = ['pearson_coeff','p-value'], columns = ['resultat_test'])


#la p-value < 5%, le coefficient est proche de 1, il y a une corrélation entre les deux variables.
#Lorsque le nombre de view augmentent les transactions augmentent aussi.

#on va se tourner vers une modélisation de regression linéaire (prevoir le nb de transactions en fonction du nb de view)


# In[58]:


from sklearn.cluster import KMeans 

from scipy.spatial.distance import cdist

liste=[2,3,4,5,6]

distortions=[]

for k in liste:
    clf=KMeans(n_clusters=k)
    clf.fit(visitor_transaction)
    distortions.append(sum(np.min(cdist(visitor_transaction, clf.cluster_centers_, 'euclidean'), axis=1)) / np.size(visitor_transaction, axis = 0)) #ici on appliquer la formule de la distortion


# In[59]:


plt.plot(liste, distortions)
plt.xlabel('Nombre de Clusters K')
plt.ylabel('Distortion (WSS/TSS)')
plt.title('Méthode du coude affichant le nombre de clusters optimal')
plt.show()


# In[34]:


visitor_transaction.sort_values(by='nb_transac',ascending=False)


# In[30]:


verif=df.loc[(df['visitorid']==152963) & (df['itemid']==119736)]
verif

verif1=df.loc[(df['visitorid']==1172087) & (df['itemid']==312728)]
verif1

verif2=df.loc[(df['visitorid']==1407512) & (df['itemid']==54141)]
verif2

#pas de ligne ajout au panier


# In[15]:


visit_view1 = df.loc[df['event'] == 'view'].groupby(by=['visitorid'], as_index=False).agg({'event':'count','visitorid':'count'})

visit_view1 = visit_view.rename(columns={"event": "nb_view"})

visit_view1


# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


# Le lien entre panier et transaction 
# On a 43% de transactions sur les paniers pas besoin de faire un test on regarde la proportion

#NAN peut pas faire de test 
#remplacer les ids par 1 et NaN par zéro 

#Ecart de temps entre temps de mise au panier et achat 

# Au bout de combien de vus, paniers, ils achètent ? 

#Création d'un nouveau data frame pour faire des test, des modèles 

#Qui sont les doublons ? 


# In[52]:


df1=df

df1['transactionid']=df1['transactionid'].replace([0.0],[1])

df1['transactionid'].value_counts()


# In[53]:


df1['transactionid']=df1['transactionid'].fillna(0)

df1['transactionid'].value_counts()


# In[54]:


df1['transactionid']=df1['transactionid'].where(df1['transactionid'] <= 0, 1)


# In[55]:


df1['event']=df1['event'].replace(['view','addtocart', 'transaction'],[1,2,3])
df1['event'].astype(str)
df1['event'].value_counts()


# In[56]:


# J'ai toutes mes variables catégorielles pour tester un modèle

ML=df1.groupby(by=['itemid', 'event','month','day','hour','transactionid'], as_index=False).size()

ML


# In[57]:


# Tester avec un nouveau dataframe

ML1=df1.groupby(by=['itemid','month','day','hour','transactionid'], as_index=False).agg({'event':'count'})

ML1


# In[9]:


# Prédire transaction en fonction des variables objets, size (le nb de fois qu'apparaît l'objet, les dates )

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# In[10]:


target=ML['transactionid']

data=ML.drop('transactionid', axis=1)


# In[11]:


X_train, X_test, y_train, y_test=train_test_split(data,target, test_size=0.2, random_state=123)

tree=DecisionTreeClassifier(criterion = "entropy", max_depth = 4 , random_state=123)

tree.fit(X_train, y_train)

y_pred=tree.predict(X_test)


# In[12]:


pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# In[13]:


test_gini=DecisionTreeClassifier(criterion = "gini", max_depth = 4 , random_state=321)
test_gini.fit(X_train, y_train)

y_pred=test_gini.predict(X_test)

pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# In[ ]:





# In[14]:


feats = {}
for feature, importance in zip(data.columns, test_gini.feature_importances_):
    feats[feature] = importance 
    
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'})
importances.sort_values(by='Importance', ascending = False ).head(3)


# In[ ]:




