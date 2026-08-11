# Brief visuel — Application de surveillance par reconnaissance faciale

## 1. En une phrase

Application **desktop** de vidéosurveillance intelligente : elle branche une ou plusieurs caméras (webcam, caméra IP, smartphone), détecte les visages en temps réel, distingue les personnes **connues** des **inconnues**, déclenche des alertes et archive chaque événement avec sa photo.

## 2. À qui c'est destiné

Usage sécurité / domicile / petit local professionnel. L'utilisateur type surveille son espace depuis un poste fixe. Ton visuel attendu : **sérieux, technique, rassurant** — un outil de sécurité, pas un gadget. Univers « poste de contrôle / centre de supervision ».

## 3. Plateforme et contraintes techniques

- **Application de bureau** (Python / Tkinter + ttkbootstrap), pas une app web ni mobile.
- Fenêtre principale **1280 × 740 px**, redimensionnable, orientation paysage.
- Le rendu passe par des widgets natifs : privilégier des maquettes en **aplats, formes simples, coins peu ou pas arrondis**, sans ombres portées complexes ni dégradés fins (difficiles à reproduire fidèlement). Le designer peut proposer mieux, mais il faut garder l'implémentabilité en tête.
- Thème **sombre uniquement** pour l'instant.

## 4. Système de design actuel (base de départ)

Une palette est déjà définie et utilisée partout — elle sert de socle, le designer peut l'affiner mais elle donne l'intention.

**Fonds (du plus profond au plus clair)**
- `#0B1120` — fond principal, noir bleuté
- `#111827` — barres et en-têtes, bleu nuit
- `#1E293B` — cartes et panneaux, slate foncé

**Texte**
- `#F8FAFC` — texte principal (blanc cassé)
- `#94A3B8` — texte secondaire, libellés, métadonnées (gris bleuté)

**Marque & accents**
- `#2563EB` — bleu électrique, action principale
- `#06B6D4` — cyan, indicateurs d'analyse / IA

**États (essentiels — code couleur sémantique fort)**
- `#22C55E` vert — accès autorisé / visage connu / flux sain
- `#F59E0B` orange — attention / reconnexion en cours
- `#EF4444` rouge — intrusion / visage inconnu / panne
- `#8B5CF6` violet — personne actuellement suivie (« cible »)

**Typographie** : Helvetica. Titres 15px bold, sous-titres 11px bold, corps 10px, petits libellés 9px, badges 8px bold. Une police mono (Courier 9px) pour les données techniques (clé d'API, etc.).

**Espacements** : échelle 2 / 4 / 8 / 12 / 16 px.

> Ces couleurs d'état sont aussi dessinées **directement sur la vidéo** : cadre vert autour d'un visage connu, rouge autour d'un inconnu, violet autour d'une personne suivie. La cohérence UI ↔ annotation vidéo est importante.

## 5. Écrans à concevoir

### A. Tableau de bord principal (écran central)
Structure actuelle, à magnifier :

- **Barre supérieure**, deux lignes :
  - Ligne 1 : logo/pastille + titre « Surveillance » · bouton **▶ Démarrer** (vert) · bouton **■ Arrêter** (rouge, désactivé au repos) · à droite, sélecteur de **Profil** (jeux de réglages).
  - Ligne 2 : navigation — **＋ Caméra · Images · Vidéos · Historique · Visages · Alertes** · à droite un bouton **API** avec pastille d'état colorée.
- **Bandeau de statut** : une ligne de message contextuel (« Prêt. Démarrez la surveillance ou ajoutez une caméra. »).
- **Corps en deux panneaux redimensionnables** :
  - **Gauche** : grille de **tuiles caméra** + liste des caméras configurées (avec boutons Modifier / Supprimer / Tester).
  - **Droite** : **journal des événements** en temps réel (cartes empilées, avec vignette photo, nom/type, horodatage, code couleur selon connu/inconnu).

### B. Tuile caméra (composant clé, à soigner)
- **En-tête** : pastille + badge d'état textuel (`EN LIGNE` vert / `CONNEXION…` orange / `RECONNEXION 5 s` orange / `HORS LIGNE` rouge) · à droite le **FPS** (cadence).
- **Corps** : le flux vidéo live (avec cadres colorés sur les visages). État vide : « Pas de signal ».
- **Pied** : nom de la caméra · nombre de visages détectés.
- Double-clic → **plein écran**.

### C. Carte d'événement (dans le journal)
Vignette du visage capturé + nom (ou « Inconnu ») + type d'événement + date/heure, liseré coloré à gauche selon la gravité (vert connu / rouge inconnu).

### D. Fenêtres secondaires (mêmes codes visuels)
- **Import d'images** — enregistrer un visage : sélection d'images, nom de la personne, aperçu, « Enregistrer le visage ».
- **Historique des détections** — tableau filtrable (Date/Heure, Caméra, Personne, Type), pagination (◀ Précédente / Suivante ▶), détail, nettoyage.
- **Configuration caméra** — formulaire : nom, source (Webcam locale / Caméra IP / Smartphone), résolution, modèle de détection (HOG rapide / CNN précis), zone d'intérêt optionnelle.
- **Configuration des alertes** — cases à cocher : alerter sur visage connu/inconnu, notifications bureau, email, webhook, anti-spam en secondes.
- **Accès à l'API** — affichage d'une clé, bouton « Copier la clé ».

## 6. États et micro-interactions à ne pas oublier

- **Système à l'arrêt vs en marche** (Démarrer/Arrêter changent tout le contexte).
- **Caméra** : en ligne / connexion / reconnexion (avec compte à rebours) / hors ligne.
- **Détection** : visage connu (vert) / inconnu (rouge) / personne suivie (violet).
- **Vides** : aucune caméra configurée, aucun événement, pas de snapshot, pas de signal.
- **API** : active / inactive (pastille).

## 7. Livrables attendus du designer

1. Une **maquette haute-fidélité** du tableau de bord (état « en marche », plusieurs caméras, journal rempli).
2. Le **composant tuile caméra** dans ses 4 états.
3. Le **composant carte d'événement** (connu / inconnu).
4. Les **fenêtres secondaires** au moins en gabarit.
5. Un **système** propre : palette affinée, typo, iconographie (états, caméra, personne, alerte), espacements — livrable idéalement en composants réutilisables (Figma).
