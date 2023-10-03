import pandas as pd
import numpy as np


df1 = pd.read_excel('BDD_NAO.xlsx', sheet_name='Echantillon NAO 2023')
df2 = pd.read_excel('BDD_NAO.xlsx', sheet_name='Saisie')
df3 = pd.merge(df1, df2, how='right', left_on=['Entreprise'], right_on=["Nom de l'entreprise "])
df4 = df3[['UrlLegifrance', 'Entreprise',
        # 'recherche v', 
      #   'Siret_x',
    #    'Secteur SECAFI', 
       'Secteur', 
    #    'Naf732', 'Date Texte',
    #    'Tranche Effectif(Base siren)', 'Nature', 'Titre', 'Date Maj',
    #    'Date Dépot', 'Date diffusion', 'Date fin', 
    #    'LesSyndicats', 'LesThemes',
       'Fichier', 
    #    'Unnamed: 18', 
       "Nom de l'entreprise ", 
   #     'Siret_y',
    #    'Sous secteur Secafi ', 'Secteur NAF ', 'Code NAF',
    #    "Date de l'accord du texte", 'Année d'application NAO',
    #    'Tranche Effectif (Base siren)',
       'Augmentations générales-Toutes catégories confondues ',
       'Augmentations générales-Cadres/ingénieurs',
       'Augmentations générales-Professions intermédiaires (techniciens, agents de maitrise, ou autres) ',
       'Augmentations générales-Ouviers, employés ou autres ',
       'Augmentations générales-Talon (€)',
       'Augmentations individuelles-Toutes catégories confondues ',
       'Augmentations individuelles-Cadres/ingénieurs',
       'Augmentations individuelles-Professions intermédiaires (techniciens, agents de maitrise, ou autres) ',
       'Augmentations individuelles-Ouviers, employés ou autres ',
       'Augmentations individuelles-Talon (€)',
    #    'Augmentations totales (AI + AG)-Toutes catégories confondues ',
    #    'Augmentations totales (AI + AG)-Cadres/ingénieurs',
    #    'Augmentations totales (AI + AG)-Professions intermédiaires (techniciens, agents de maitrise, ou autres) ',
    #    'Augmentations totales (AI + AG)-Ouviers, employés ou autres ',
    #    'Augmentations totales (AI + AG)-Talon (€)', 'Primes -Pepa ou autres ',
       'Primes Pepa ou autres-Montant en € ',
    #    'Elements exceptionnels liés à l'inflation ? Prime transport ou autre ',
    #    'Commentaires-Divers', 'Lien '
    ]].copy()

df4.to_excel('matching.xlsx', index=False)
df4['Num_Fichier']= df4['Fichier'].str.slice(101, 128)
df4.dropna().reset_index()
df5 = pd.read_excel('processed_df_V2.2.xlsx')
df_match = pd.merge(df4, df5, how='outer', left_on='Num_Fichier', right_on="Fichiers")
df_match.to_excel('checking.xlsx', index=False)