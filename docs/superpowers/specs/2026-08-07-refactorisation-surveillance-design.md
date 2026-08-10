# Refactorisation du système de surveillance — design

**Date :** 2026-08-07
**Statut :** validé
**Périmètre :** `src/face_recognition_app/` dans son intégralité

---

## 1. Contexte

Le projet est une application de surveillance domestique par reconnaissance faciale
(5 820 lignes, 20 modules, Python 3.12, Tkinter + OpenCV + dlib + SQLite + Flask).

Une analyse complète a établi son état :

- L'architecture en couches est saine, mais **deux générations de code cohabitent**.
  Une v1 prototype (`interface.py`, `video_importer.py`, `import_image.py`,
  `core/utils.py`, ~1 500 lignes) duplique ce que la v2 structurée
  (`services/`, `storage/`, `surveillance_dashboard.py`) fait mieux.
- **12 bugs confirmés**, dont 4 bloquants sous Linux.
- **L'API REST est ouverte sur `0.0.0.0:5000` sans aucune authentification**, exposant
  les flux caméra, l'historique nominatif et l'arrêt de la surveillance.
- Les **profils de surveillance ne sont appliqués qu'à moitié** : 8 champs sur 13 sont
  du code mort, dont `record_video`.
- **1 460 lignes de services n'ont aucun test.** Les 14 tests existants couvrent
  2 modules sur 20.

## 2. Objectifs

Le système doit devenir **réellement utilisable en continu au domicile de
l'utilisateur**. Cela fixe trois exigences fermes :

1. **Sécurité** — aucun accès non authentifié aux flux ni aux données biométriques.
2. **Fiabilité** — une caméra qui perd son flux doit se reconnecter, toujours.
3. **Stabilité dans la durée** — pas de fuite mémoire, de connexion ou de disque sur
   plusieurs jours de fonctionnement.

À quoi s'ajoute l'objectif structurel : **une seule génération de code**, avec des
services totalement indépendants de l'UI.

## 3. Décisions de cadrage

| Question | Décision |
|---|---|
| Finalité | Usage réel au domicile. Sécurité et robustesse 24/7 sont des exigences fermes. |
| Sort de la couche v1 | **Supprimée.** Le dashboard devient l'unique application. L'alarme sonore sur personne ciblée — seule fonctionnalité propre à la v1 — migre vers un canal d'`AlertManager`. |
| Import vidéo | **Conservé, réécrit.** La logique part dans un service `face_extraction.py` testable, avec regroupement vectorisé au lieu de l'actuel O(n²). |
| Plateforme UI | **Tkinter maintenant, web ensuite.** On refactorise pour que les services soient strictement UI-agnostiques, afin qu'une UI web puisse être ajoutée plus tard sans retoucher le métier. |
| Stratégie | **Par couche, filet de tests d'abord.** Socle → storage → services → UI, corrections intégrées à chaque couche. |

Le choix de la stratégie mérite sa justification : sur les 12 bugs, 5 se situent dans
du code destiné à disparaître. Les corriger avant de refactoriser serait du travail
perdu. Et le risque réel du projet n'est pas la lenteur mais la régression sur un
moteur de surveillance dépourvu de tests.

## 4. Architecture cible

Principe directeur : **les services ne connaissent jamais l'UI.** C'est ce qui rend
l'UI web ultérieure possible sans retoucher le métier. Ce principe est aujourd'hui
violé — le dashboard écrit lui-même en base à la réception d'un événement.

```
src/face_recognition_app/
├── __main__.py            # composition root : config, logging, câblage, lancement
├── settings.py            # AppSettings (dataclass) — chemins, seuils, secrets via env
│
├── domain/                # modèles purs — zéro I/O, zéro dépendance externe
│   ├── camera.py          # CameraConfig
│   ├── detection.py       # DetectedFace, SurveillanceEvent
│   └── profile.py         # SurveillanceProfile ← source unique de vérité
│
├── storage/               # persistance derrière des interfaces explicites
│   ├── encodings_repository.py
│   ├── event_repository.py     # SQLite + thread d'écriture + purge auto
│   ├── profile_repository.py
│   └── camera_repository.py
│
├── services/              # métier, strictement UI-agnostique
│   ├── camera_source.py
│   ├── camera_manager.py
│   ├── motion_detector.py
│   ├── face_matcher.py
│   ├── recognition_engine.py
│   ├── event_bus.py
│   ├── event_recorder.py       # NOUVEAU : écoute le moteur, persiste les événements
│   ├── video_recorder.py
│   ├── alert_manager.py
│   ├── channels/               # desktop.py, email.py, webhook.py, sound.py
│   └── face_extraction.py      # enrôlement depuis images ou vidéo
│
├── api/
│   ├── server.py
│   ├── auth.py
│   └── routes.py
│
└── ui/                    # présentation uniquement
    ├── dashboard.py            # ~250 lignes au lieu de 782
    ├── widgets/                # camera_tile.py, camera_grid.py, event_log.py
    └── dialogs/                # camera_config, alerts_config, event_browser, enrollment
```

### 4.1 Injection de dépendances, fin des globales

`storage/config.py` expose aujourd'hui `ENCODED_DIR` comme constante de module, au
point que les tests doivent la muter (`store.ENCODED_DIR = str(tmp_path)`). Elle est
en outre dérivée de `Path.cwd()`, ce qui fait dépendre la base d'encodages utilisée du
répertoire de lancement.

Un objet `AppSettings` est construit une fois dans `__main__` et passé explicitement
aux composants. Aucun module ne lit de configuration à l'import.

### 4.2 `SurveillanceProfile` devient la seule source de vérité

`AlertConfig` duplique aujourd'hui `alert_on_unknown`, `alert_on_known` et
`target_persons`, persistés dans deux fichiers distincts — et seule la version
`AlertConfig` a un effet, celle du profil étant du code mort.

Les deux sont fusionnés dans `SurveillanceProfile`. `AlertConfig` ne conserve que ce
qui relève du transport (serveur SMTP, URL de webhook, activation des canaux). Le
moteur applique **l'intégralité** du profil.

### 4.3 `EventRecorder` s'intercale entre le moteur et l'UI

Le dashboard ne persiste plus rien. Un service s'abonne au moteur pour enregistrer,
l'UI s'y abonne séparément pour afficher. C'est ce découplage qui permettra à une UI
web de s'abonner de la même façon.

### 4.4 Domaine testable sans matériel

Aucun import d'OpenCV, de dlib ni de Tkinter dans `domain/`. La logique de décision
(faut-il alerter ? enregistrer ? ce visage correspond-il ?) devient du calcul pur.

### 4.5 Suppressions

`interface.py`, `video_importer.py`, `import_image.py`, `core/utils.py` — ~1 500
lignes. L'alarme sonore migre vers `services/channels/sound.py`, le gestionnaire
d'encodages devient un dialogue autonome, l'import vidéo un service.

`main.py` est conservé à la racine comme lanceur local (il ajoute `src/` au
`PYTHONPATH` et appelle `__main__.main()`), mais le menu à six boutons qu'il ouvrait
disparaît : le dashboard devient la fenêtre de démarrage.

Code mort à retirer dans le même mouvement : `CameraStats._fps_tick` (stub `pass`),
`IPCameraSource` (classe `pass` sans spécialisation RTSP — elle sera soit dotée d'un
vrai comportement, backend FFMPEG et `CAP_PROP_BUFFERSIZE=1`, soit supprimée au profit
d'un paramètre sur la source générique), `config.CAMERAS_FILE` (redéfini dans le
dashboard), `progress_queue`/`check_progress` (file jamais alimentée),
`MotionDetector.get_motion_mask`/`set_roi`/`reset`, `EventStore.get_for_date`,
`SurveillanceProfile.enabled_camera_uids`, `interface.new_faces`/`is_capturing`.

## 5. Flux de données et modèle de threads

### 5.1 Bus d'événements

Le moteur appelle aujourd'hui ses écouteurs en synchrone depuis le thread d'analyse
(`surveillance_engine._emit`). Comme le dashboard y encode un JPEG et écrit en SQLite
sous verrou global, la reconnaissance s'interrompt pendant ce temps ; avec plusieurs
caméras détectant simultanément, les threads d'analyse se sérialisent.

```
Thread d'analyse ──publish()──► EventBus (file bornée, non bloquant, drop si pleine)
                                     │
                                     ▼  thread dispatcher unique
                     ┌───────────────┼────────────────┬──────────────┐
                     ▼               ▼                ▼              ▼
              EventRecorder    AlertManager    VideoRecorder      UI
             (file d'écriture   (thread par     (si profil       (file drainée
              SQLite dédiée)     envoi)         record_video)     par after())
```

`publish()` est instantané : le thread d'analyse n'attend jamais un consommateur lent.
La file du bus est bornée ; en saturation, les événements les plus anciens sont
abandonnés et le fait est journalisé.

### 5.2 Threads par caméra

| Thread | Rôle | Cadence |
|---|---|---|
| `cam-<uid>` | lecture du flux, machine à états de reconnexion | au fil du flux |
| `prod-<uid>` | prélève une frame, alimente le buffer d'enregistrement | `settings.capture_fps` |
| `surv-<uid>` | mouvement → reconnaissance → `publish()` | `profile.analysis_interval` |

Plus trois threads partagés : dispatcher du bus, writer SQLite, serveur API.

**Tous les `time.sleep` deviennent des `Event.wait()`.** C'est ce qui rend `stop()`
réellement immédiat : aujourd'hui un `time.sleep(60)` dans la boucle de reconnexion
fait échouer le `join(timeout=3)`, laissant le thread manipuler une capture déjà
libérée.

### 5.3 Reconnexion : machine à états explicite

```
DISCONNECTED ──tentative──► CONNECTING ──succès──► CONNECTED
      ▲                          │                     │
      └──échec, backoff 2→60 s───┘◄────perte de flux───┘
```

L'état `DISCONNECTED` retente **toujours**. C'est précisément ce qui manque
aujourd'hui : après un échec de réouverture, `_cap` reste `None` et la boucle prend
définitivement la branche inerte.

L'état est exposé en lecture, pour que la vignette affiche « reconnexion dans 8 s »
plutôt qu'un simple point rouge.

### 5.4 Corrections de cadence

- **Le profil pilote la chaîne.** Le producteur n'alimente le buffer d'enregistrement
  que si `profile.record_video` est vrai. Aujourd'hui il copie 30 frames/s/caméra en
  mémoire même en profil `present`, qui n'enregistre rien.
- **Une seule définition des fps.** `settings.capture_fps` sert au producteur *et* au
  `VideoRecorder`. Fin des clips en vitesse ×2 et du buffer « 5 s » n'en contenant
  que 2,5.
- **L'UI se rafraîchit sur événement.** Une vignette n'est redessinée que si une
  nouvelle frame est arrivée ; l'intervalle passe de 100 ms à 200 ms.

### 5.5 Cycle de vie

`start()` et `stop()` deviennent idempotents. `start()` n'enregistre son callback
`on_change` qu'une fois ; `stop()` le retire et purge détecteurs, stats, files et
threads. Supprimer une caméra libère aussi son buffer d'enregistrement.

## 6. Sécurité

### 6.1 API : fermée par défaut, ouverture explicite

- `settings.api_host` vaut `127.0.0.1`. L'exposition réseau devient une case à cocher
  dans l'UI, assortie d'un avertissement.
- **Toute route exige `X-API-Key`**, comparée avec `hmac.compare_digest`. La clé vient
  de `FR_API_KEY` si définie, sinon elle est générée au premier lancement
  (`secrets.token_urlsafe(32)`) et écrite en `0600` dans le répertoire de
  configuration. Le dashboard l'affiche avec un bouton « copier ».
- **Lecture et écriture séparées.** Les routes d'écriture (`/api/surveillance/start`
  et `/stop`) sont gouvernées par `settings.api_allow_control`, à `false` par défaut :
  consulter ses caméras à distance n'oblige plus à laisser la télécommande ouverte.
- **Limitation des tentatives** : blocage d'une IP après 10 échecs d'authentification
  en 5 minutes.
- Le serveur utilise `werkzeug.serving.make_server` afin que `stop()` arrête
  réellement l'écoute.

### 6.2 Secrets et données personnelles

- Le mot de passe SMTP quitte le JSON : lu depuis `FR_SMTP_PASSWORD`, avec le
  trousseau système en option si `keyring` est disponible.
- **Migration** : si un `alerts_config.json` existant contient un mot de passe en
  clair, il en est retiré au démarrage et l'utilisateur est averti de le repositionner
  en variable d'environnement.
- Encodages faciaux et `events.db` reçoivent des permissions `0600` à la création. Le
  README documente leur nature biométrique et leur emplacement.
- Les caractères `%` et `_` d'un nom sont échappés avant la requête `LIKE` de
  recherche par personne.

## 7. Politique d'erreurs

Le code compte une vingtaine de `except Exception: pass`, dont trois consécutifs dans
le moteur. Un enregistrement vidéo qui échoue est aujourd'hui parfaitement silencieux.

- **Rien n'est avalé sans trace.** Toute exception est journalisée avec son contexte
  (caméra, opération). Un `pass` nu ne subsiste que là où l'échec est sans
  conséquence, et porte alors un commentaire qui le justifie.
- **L'isolation des pannes est explicite.** Un écouteur qui lève ne tue pas le
  dispatcher ; une caméra en échec n'arrête pas les autres ; un canal d'alerte cassé
  n'empêche pas les trois autres de partir.
- **Les échecs de démarrage sont bruyants.** Base illisible, répertoire d'encodages
  inaccessible, clé API non générable : message clair et arrêt, plutôt qu'un système
  qui tourne à moitié en croyant surveiller.

## 8. Tenue dans la durée

Trois garde-fous entrent dans `AppSettings` :

- `event_retention_days` (**défaut 0 = conservation illimitée**). Voir la révision
  ci-dessous : ce point a été corrigé en cours d'implémentation.
- `max_clips` et un plafond en volume sur `clips/`.
- Les connexions SQLite par thread sont fermées à l'arrêt du thread, au lieu de fuir à
  chaque cycle start/stop.

Tout ce qui est spécifique à Windows disparaît : plein écran (`state("zoomed")`),
filtres de fichiers à points-virgules. Un contrôle au démarrage signale l'installation
simultanée de `opencv-python` et `opencv-python-headless`, qui casse silencieusement
l'affichage OpenCV.

## 9. Stratégie de tests

On ne fige pas le comportement *actuel* mais le comportement *voulu*. Pour les 12
bugs, le test échoue d'abord, puis la correction le fait passer.

| Niveau | Objet | Moyen |
|---|---|---|
| `domain/` | Décisions pures : alerter ? enregistrer ? correspondance ? | Calcul pur |
| `storage/` | Repositories, purge, migration, échappement | `tmp_path`, SQLite en mémoire |
| `services/` | Reconnexion, mouvement, moteur, bus, cadences, canaux | `FakeCameraSource`, frames numpy synthétiques |
| `api/` | Auth absente / invalide / valide, limitation, lecture-écriture | `app.test_client()` |

**La reconnaissance faciale est mockée à la frontière `face_matcher`.** Aucun test ne
charge dlib, n'ouvre de caméra ni n'exige un écran. La suite doit rester sous les
10 secondes.

L'UI Tkinter reste testée manuellement : le rapport coût/bénéfice d'un harnais Tk ne
le justifie pas, et vider la logique métier hors de l'UI fait qu'il n'y reste presque
plus rien à tester.

**Cible : ~80 % sur tout ce qui n'est pas UI.**

### Outillage

- `[tool.pytest.ini_options]` avec `pythonpath = ["src"]`, supprimant le bricolage de
  `sys.path` dans `conftest.py`.
- **ruff** (lint + format), **mypy** sur `domain/` et `storage/` uniquement.
- **CI GitHub Actions** : ruff + pytest à chaque push.
- Versions épinglées, `requirements-dev.txt` séparé, `playsound` et `psutil` retirés
  (déclarés mais jamais utilisés ; `playsound` n'est même pas installé).

## 10. Lots de livraison

Chaque lot se termine avec une suite verte et constitue un commit cohérent.

| Lot | Contenu | Bugs traités |
|---|---|---|
| **0** | Outillage, CI, pins, filet de caractérisation | — |
| **1** | `settings.py`, `domain/`, logging centralisé, fin des globales | ⑫ |
| **2** | Repositories, purge auto, thread d'écriture, connexions fermées | — |
| **3** | Sources caméra : machine à états, `Event.wait`, cycle de vie | ③ |
| **4** | Moteur + bus + `EventRecorder`, profil appliqué en entier | ⑤ ⑥ ⑦ |
| **5** | Enregistrement (fps unifiés) + alertes (canal son, profil unique, secrets) | ⑤ ⑧ |
| **6** | API : auth, `127.0.0.1`, lecture/écriture séparées, arrêt réel | ④ |
| **7** | UI : dashboard découpé, v1 supprimée, enrôlement vidéo réécrit | ① ② ⑨ ⑩ ⑪ |
| **8** | README, documentation, code mort, `.gitignore`, vérification OpenCV | — |

Précisions sur le lot 8 : le `.gitignore` actuel ignore `*.json` dans tout le dépôt,
règle trop large qui masquerait tout fichier de configuration versionné à l'avenir.
Elle est remplacée par des règles ciblées (`cameras.json`, `profiles.json`,
`alerts_config.json`, `encodings/*.json`). Le README est réécrit pour refléter
l'application unique, documenter les variables d'environnement (`FR_API_KEY`,
`FR_SMTP_PASSWORD`) et la nature biométrique des données stockées.

Le lot 6 ne dépend que du lot 1 : il peut être remonté juste après le socle si l'on
veut sécuriser le système au plus vite.

## 11. Registre des bugs

| # | Description | Emplacement actuel | Lot |
|---|---|---|---|
| ① | `state("zoomed")` inexistant sous Linux — le plein écran lève `TclError` | `surveillance_dashboard.py:569` | 7 |
| ② | « Import vidéos » détourne la fenêtre du dashboard puis lève `AttributeError` | `surveillance_dashboard.py:620` | 7 |
| ③ | Caméra jamais reconnectée si la première tentative échoue | `camera_source.py:181-208` | 3 |
| ④ | API impossible à réactiver ; `stop()` n'arrête pas le serveur | `api_server.py:200` | 6 |
| ⑤ | Profils appliqués à moitié : 8 champs sur 13 sans effet | `surveillance_engine.py:138-147` | 4, 5 |
| ⑥ | Sensibilité mouvement du profil ignorée par les caméras ajoutées ensuite | `surveillance_engine.py:200` | 4 |
| ⑦ | Fuite de callbacks `on_change` à chaque redémarrage | `surveillance_engine.py:171` | 4 |
| ⑧ | Alarme jamais redéclenchée après un arrêt manuel | `interface.py:798-805` | 5 |
| ⑨ | Appels Tkinter depuis un thread de travail | `video_importer.py:482-484` | 7 |
| ⑩ | Miniatures de groupes vides (référence `PhotoImage` perdue) | `video_importer.py:354-356` | 7 |
| ⑪ | Filtres de fichiers à points-virgules, inopérants sous Linux | `interface.py:295` et 2 autres | 7 |
| ⑫ | `opencv-python` et `opencv-python-headless` installés ensemble | environnement | 1 |

## 12. Hors périmètre

- **L'UI web.** Le design prépare ses frontières mais ne l'implémente pas. Elle fera
  l'objet d'un projet distinct.
- **Le remplacement de dlib** par un modèle plus récent (InsightFace, MediaPipe).
  L'abstraction `face_matcher` rendra cette substitution possible, mais elle n'est pas
  entreprise ici.
- **Le multi-utilisateur, les rôles, le chiffrement au repos.** Hors sujet pour un
  système mono-foyer.
- **La détection d'objets ou de mouvement avancée** (personnes, véhicules, animaux).

## 12 bis. Révision du 2026-08-10 — la purge ne doit rien supprimer d'elle-même

**Ce que la spec disait initialement.** `event_retention_days` valait 30 par défaut,
avec purge automatique au démarrage puis quotidienne, et `VACUUM` périodique.

**Ce qui s'est passé.** Pendant l'implémentation de la Phase 1, une étape de
vérification a appelé `build_context()` sur le répertoire réel du projet. La purge
s'est exécutée comme spécifié et a supprimé **715 événements** datant du 7 mai, soit
94 jours — au-delà de la rétention. Le `VACUUM` qui a suivi a rendu la perte
irrécupérable : pas de `-wal`, pas de sauvegarde, et `events.db` est dans
`.gitignore` donc absent de l'historique.

**Pourquoi c'était un défaut de conception, pas seulement une erreur d'exécution.**
Si la purge ne s'était pas déclenchée là, elle se serait déclenchée au premier
lancement réel chez l'utilisateur, avec le même résultat. Un système de surveillance
dont la raison d'être est de conserver un historique de détections ne peut pas
l'effacer de lui-même, silencieusement, au démarrage.

**Décision.** La rétention passe à `0` par défaut, ce qui signifie « conserver
indéfiniment » :

- `purge_expired()` retourne immédiatement `0` tant que `event_retention_days <= 0`.
- Lorsqu'une rétention est configurée, une sauvegarde `events.db.bak` précède
  toute suppression effective, réalisée via l'API `Connection.backup()` de SQLite.
- Le nombre de lignes visées est journalisé en `WARNING` **avant** l'opération.
- Aucune purge n'a lieu au démarrage. Le nettoyage devient une action manuelle,
  déclenchée depuis la fenêtre d'historique, avec confirmation.
- Les vérifications des plans d'implémentation s'exécutent sur une copie temporaire
  des données, jamais sur la racine du projet.

`max_clips` conserve en revanche sa suppression automatique : un clip vidéo est
volumineux et reconstituable par une nouvelle détection, contrairement à une ligne
d'historique.

## 13. Compatibilité des données

La refactorisation **préserve les données existantes** : les 4 encodages de
`encodings/`, les 8,5 Mo d'`events.db`, `cameras.json` et `profiles.json` restent
lisibles sans intervention manuelle.

Là où un format évolue — fusion d'`AlertConfig` dans `SurveillanceProfile`, retrait du
mot de passe SMTP — une migration au démarrage convertit l'ancien format et journalise
ce qu'elle a fait. Le schéma SQLite `events` reste inchangé.
