
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from datetime import time
from streamlit_searchbox import st_searchbox


import streamlit as st



st.header('',divider='rainbow')
texte_a_centrer = "Creuse Ta Base"
# Utilisation de la balise HTML <div> pour centrer le texte
st.markdown(
    f"<div style='text-align: center;'><h1>{texte_a_centrer}</h1></div>",
    unsafe_allow_html=True
)
st.header('',divider='rainbow')
st.write("")

st.markdown("Bienvenue sur notre interface de recommendation de films, à l'aide de la barre de recherche ci-dessous, vous pourrez taper le nom d'un film que vous aimez et s'en suivra 10 recommendations :sunglasses:")
st.write("")
st.write('Vous avez oublié le nom de votre film ? Servez-vous des filtres sur la gauche !')
df_final = pd.read_csv(r'/Users/swell/Documents/test_streamlit/df_final_ML.csv')
df_final.drop(columns = 'Unnamed: 0', inplace = True )

url = pd.read_csv(r'/Users/swell/Documents/test_streamlit/df_hugo_max-2.csv') # Final TMBD
acteur_realisateur = pd.read_csv(r'/Users/swell/Documents/test_streamlit/acteur_realisateur.csv') # Personnes


# Remplacer les valeurs NaN dans la colonne 'poster_path' par une chaîne vide
url['poster_path'].fillna('', inplace=True)

#url[url['genres'] == Genre]

#recherche centrale





def search_titles(search_term):
    return url[url['title'].str.contains(search_term, case=False)]['title'].tolist()

# Utiliser la boîte de recherche Streamlit
selected_title = st_searchbox(
    search_titles,
    placeholder="Rechercher par titre...",
    key="title_searchbox")
st.write("")

# Remplacer les valeurs NaN dans la colonne 'title' par une chaîne vide
# Filtrer le DataFrame en fonction de la recherche IMDb
if selected_title: 
 
    filtered_data = url[url['title']== selected_title]['tconst'].values[0]


    cible = df_final[df_final['tconst']== filtered_data]



################################### Machine Learning ###################################



    X = df_final[['nconst_0',
        'nconst_1', 'nconst_2', 'nconst_3', 'nconst_4', 'nconst_5', 'nconst_6',
        'nconst_7', 'nconst_8', 'nconst_9', 'numVotes','ponderation','Action', 'Adult', 'Adventure', 'Animation', 'Biography',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
        'Romance', 'Sci-Fi', 'Science Fiction', 'Sport', 'Thriller', 'War',
        'Western', 'decade' ]]



    # Create and fit a scaler model
    scaler = StandardScaler().fit(X)

    # Your scaler model can now transform your data
    X_scaled = scaler.transform(X)

    # Création des clusters
    Model_KNN = KMeans(n_clusters=5, random_state=42)
    Model_KNN.fit(X_scaled)
    df_final['Cluster_Labels'] = Model_KNN.predict(X_scaled)




    # Extraire les films du même cluster
    films_cluster = df_final[df_final['Cluster_Labels'] == cible['Cluster_Labels'].values[0]]

    X = films_cluster[[ 'nconst_0', 'nconst_1', 'nconst_2', 'nconst_3', 'nconst_4', 'nconst_5', 'nconst_6',
        'nconst_7', 'nconst_8', 'nconst_9', 'numVotes','ponderation','Action', 'Adult', 'Adventure', 'Animation', 'Biography',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
        'Romance', 'Sci-Fi', 'Science Fiction', 'Sport', 'Thriller', 'War',
        'Western', 'decade' ]]

    knn_model = NearestNeighbors(n_neighbors=11).fit(X)

    # Create and fit a scaler model
    scaler = StandardScaler().fit(X)


    # Your scaler model can now transform your data
    X_scaled = scaler.transform(X)

    knn_model.fit(X_scaled)
    cible_scaled = scaler.transform(cible.drop(columns=['tconst', 'startYear', 'primaryTitle', 'runtimeMinutes', 'Cluster_Labels', 'averageRating']))
    voisins = knn_model.kneighbors(cible_scaled)

    liste_tconst = []
    for i in range(11):
        liste_tconst.append(films_cluster.iloc[voisins[1][0][i]][0])

    df_reco = url[url['tconst'].isin(liste_tconst)]
    df_reco = df_reco.set_index('tconst').loc[liste_tconst].reset_index()


    for index, row in df_reco.iterrows():
            colonne_image, colonne_texte = st.columns([1, 2])
            with st.container():
                with colonne_image:
                    base_url = 'https://image.tmdb.org/t/p/w1280'
                    url_utilisateur = row['poster_path']

                    if not pd.isna(url_utilisateur):
                        url_final = base_url + url_utilisateur
                        st.image(url_final, caption=row['title'], width=200)
                    else:
                        st.image('white_page.jpg', caption="No image available", width=200)
            


                st.write("")
                st.write("")



                with colonne_texte:
                    if row['ponderation'] >= 1.957083e-05 and row['averageRating'] >= 8:
                        st.markdown(f"## {row['title']} :star:", unsafe_allow_html=False)
                    else: 
                        st.markdown(f"## {row['title']} ", unsafe_allow_html=False)
                    st.write(f'**Année de sortie :** {round(row['startYear'])}')
                    if row['numVotes'] > 1000 :
                        st.write(f'**Note :** {row['averageRating']} sur {row['numVotes']//1000}k votes')
                    elif row['numVotes'] == 1 :
                        st.write(f'**Note :** {row['averageRating']} sur {row['numVotes']} vote')
                    elif row['numVotes'] == 0 :
                        st.write(f"**Note :** Pas d'information")
                    else :
                        st.write(f'**Note :** {row['averageRating']} sur {row['numVotes']} votes')
                        st.write(f'**Durée :** {round(row['runtimeMinutes'])} Minutes')


                    if not pd.isna(row['genres_x2']):
                        st.write(f'**Genres :** {row["genres_x0"]}, {row["genres_x1"]}, {row["genres_x2"]}')
                    elif not pd.isna(row['genres_x1']):
                        st.write(f'**Genres :** {row["genres_x0"]}, {row["genres_x1"]}')
                    elif not pd.isna(row['genres_x0']):
                        st.write(f'**Genres :** {row["genres_x0"]}')
                
                    
                        # Afficher les acteurs et réalisateurs

                    table_film = acteur_realisateur[acteur_realisateur['tconst']== row['tconst']]

                    directeur, acteur = '', ''

                    for loop in range(len(table_film)): 
                        if table_film['category'].values[loop] == 'actor' or table_film['category'].values[loop] == 'actress':
                            acteur = acteur + (table_film['primaryName'].values[loop]) + ', '

                        if table_film['category'].values[loop] == 'director':
                            directeur = directeur + (table_film['primaryName'].values[loop]) + ', '

                    if len(acteur) > 0:
                        st.write(f'**Casting principial :** {acteur[:-2]}')
                    else : 
                        st.write("**Casting principial :** pas d'informations")

                    if len(directeur) > 0:
                        st.write(f'**Réalisateurs :** {directeur[:-2]}')
                    else :
                        st.write(f"**Réalisateurs :** pas d'information")            
        
                    






















####################################################################### separation centrale et laterale


#Initialisation des boutons et des barres de recherches
st.sidebar.title("Filtres")
filtered_data_search_sidebar = url.copy()

search_imdb_sidebar = st.sidebar.text_input('Recherchez un film :')

selected_years = st.sidebar.slider('Sélectionnez une plage d\'années :', min_value=1951, max_value=2023, value=(1950, 2023))


genres = (' ',  'Action', 'Animation',  'Aventure', 'Biographie','Comédie', 'Crime', 'Drame', 'Documentaire', 'Famille', 'Fantaisie','Film Noir', 'Guerre', 'Histoire','Horreur',  'Musique', 'Mystère', 'Romance', 'Science-Fiction','Sport', 'Thriller',   'Western',)
selected_genre = st.sidebar.selectbox('Sélectionnez un genre :', genres)

search_actor = st.sidebar.text_input('Rechercher un membre du casting :')



if not selected_title: 
    # recherche de nom de film

    if search_imdb_sidebar and len(filtered_data_search_sidebar) > 0:
        filtered_data_search_sidebar = url[url['title'].str.contains(search_imdb_sidebar, case=False)]



    filtered_data_search_sidebar = filtered_data_search_sidebar[(filtered_data_search_sidebar['startYear'] >= selected_years[0]) & (filtered_data_search_sidebar['startYear'] <= selected_years[1])
        ].sort_values(by='startYear')


    # Filtre genres
    if selected_genre  != ' ':
        filtered_data_search_sidebar = filtered_data_search_sidebar[
            (filtered_data_search_sidebar['genres_x0'].str.contains(selected_genre, case=False)) |
            (filtered_data_search_sidebar['genres_x1'].str.contains(selected_genre, case=False)) |
            (filtered_data_search_sidebar['genres_x2'].str.contains(selected_genre, case=False))
        ]
        
    filtered_data_search_sidebar = filtered_data_search_sidebar[(filtered_data_search_sidebar['startYear'] >= selected_years[0]) & (filtered_data_search_sidebar['startYear'] <= selected_years[1])
        ].sort_values(by='startYear')

    # Filtre acteurs

    if  search_actor and len(filtered_data_search_sidebar) > 0:

        filtered_search_actor_sidebar = acteur_realisateur[
        (acteur_realisateur['primaryName'].str.contains(search_actor, case=False))] # Ajouter cette condition pour filtrer seulement les acteurs

        tconst_acteur = list(set(filtered_search_actor_sidebar['tconst']))
        filtered_data_search_sidebar = filtered_data_search_sidebar[filtered_data_search_sidebar['tconst'].isin(tconst_acteur)]

    filtered_data_search_sidebar = filtered_data_search_sidebar[(filtered_data_search_sidebar['startYear'] >= selected_years[0]) & (filtered_data_search_sidebar['startYear'] <= selected_years[1])
        ].sort_values(by='startYear')


    #Filtre par notes 




        # Filter data based on selected years within the genre




    if search_imdb_sidebar and len(filtered_data_search_sidebar) > 0 or selected_genre  != ' ' or search_actor and len(filtered_data_search_sidebar) > 0  or  selected_years[0] > 1950 or selected_years[1] < 2023: 
        
        
        for index, row in filtered_data_search_sidebar.head(50).iterrows():
            colonne_image, colonne_texte = st.columns([1, 2])
            with st.container():
                with colonne_image:
                    base_url = 'https://image.tmdb.org/t/p/w1280'
                    url_utilisateur = row['poster_path']

                    if not pd.isna(url_utilisateur):
                        url_final = base_url + url_utilisateur
                        st.image(url_final, caption=row['title'], width=230)
                    else:
                        st.image('white_page.jpg', caption="No image available", width=200)
                st.write("")
                st.write("")
                    
                with colonne_texte:

                        if row['ponderation'] >= 1.957083e-05 and row['averageRating'] >= 8:
                            st.markdown(f"## {row['title']} :star:", unsafe_allow_html=False)
                        else: 
                            st.markdown(f"## {row['title']} ", unsafe_allow_html=False)
                        

                        st.write(f'**Année de sortie :** {round(row['startYear'])}')
                        if row['numVotes'] > 1000 :
                            st.write(f'**Note :** {row['averageRating']} sur {row['numVotes']//1000}k votes')
                        elif row['numVotes'] == 1 :
                            st.write(f'**Note :** {row['averageRating']} sur {row['numVotes']} vote')
                        elif row['numVotes'] == 0 :
                            st.write(f"**Note :** Pas d'information")
                        else :
                            st.write(f'**Note :** {row['averageRating']} sur {row['numVotes']} votes')

                        st.write(f'**Durée :** {round(row['runtimeMinutes'])} minutes')
                        

                        if not pd.isna(row['genres_x2']):
                            st.write(f'**Genres :** {row["genres_x0"]}, {row["genres_x1"]}, {row["genres_x2"]}')
                        elif not pd.isna(row['genres_x1']):
                            st.write(f'**Genres :** {row["genres_x0"]}, {row["genres_x1"]}')
                        elif not pd.isna(row['genres_x0']):
                            st.write(f'**Genres :** {row["genres_x0"]}')
                        else:
                            st.write("Pas d'informations.")
                        
                            # Afficher les acteurs et réalisateurs

                        table_film = acteur_realisateur[acteur_realisateur['tconst']== row['tconst']]

                        directeur, acteur = '', ''

                        for loop in range(len(table_film)): 
                            if table_film['category'].values[loop] == 'actor' or table_film['category'].values[loop] == 'actress':
                                acteur = acteur + (table_film['primaryName'].values[loop]) + ', '

                            if table_film['category'].values[loop] == 'director':
                                directeur = directeur + (table_film['primaryName'].values[loop]) + ', '

                        if len(acteur) > 0:
                            st.write(f'**Casting principial :** {acteur[:-2]}')
                        else : 
                            st.write("**Casting principial :** pas d'informations")

                        if len(directeur) > 0:
                            st.write(f'**Réalisateurs :** {directeur[:-2]}')
                        else :
                            st.write(f"**Réalisateurs :** pas d'information")
                