from flask import Flask, render_template, request, flash, redirect, url_for, session, send_file, Response, jsonify, after_this_request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ssaw import Client, InterviewsApi, WorkspacesApi, ExportApi, QuestionnairesApi
from ssaw.models import ExportJob
from typing import List, Dict, Optional, Tuple
import json
import ssaw.exceptions
from collections import defaultdict
import base64
import requests
import uuid
import logging
import re
import tempfile
from flask_cors import CORS
from datetime import datetime

from ssaw.export import ExportApi, ExportJob
from ssaw.questionnaires import QuestionnairesApi
import ssaw.exceptions

# Autorise Angular (localhost:4200) à faire des requêtes

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200"], supports_credentials=True)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['DOWNLOAD_FOLDER'] = 'Downloads'
app.secret_key = '12fgt334jhznbkioup@sF3#Hgdgdhdj'


# Create upload and download folders if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['DOWNLOAD_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Configure logging
logging.basicConfig(filename='download.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# API POUR Recuperer les workspaces
@app.route('/api/workspaces', methods=['POST'])
def get_all_workspaces():
    data = request.get_json()

    required_fields = ['api_url', 'username', 'password']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({
            "status": "error",
            "message": f"Champs manquants: {', '.join(missing)}"
        }), 400

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=None
        )
        workspaces_api = WorkspacesApi(client)
        workspaces = workspaces_api.get_list()

        result = []
        for ws in workspaces:
            result.append({
                "name": ws.name,
                "display_name": getattr(ws, 'display_name', ''),
                "description": getattr(ws, 'description', ''),
            })

        return jsonify({
            "status": "success",
            "count": len(result),
            "data": result
        })

    except ssaw.exceptions.UnauthorizedError:
        return jsonify({
            "status": "error",
            "message": "Authentification échouée, vérifiez vos identifiants."
        }), 401
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
# API POUR Recuperer les workspaces


# API POUR Initialiser une configuration
@app.route('/api/configure', methods=['POST'])
def api_configure():
    data = request.get_json()

    required_fields = ['api_url', 'username', 'password', 'workspace']
    missing = [f for f in required_fields if f not in data]

    if missing:
        return jsonify({"status": "error", "message": f"Champs manquants: {', '.join(missing)}"}), 400

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=data['workspace']
        )

        # Test workspace
        workspaces = WorkspacesApi(client).get_list()
        names = [ws.name for ws in workspaces]
        if data['workspace'] not in names:
            return jsonify({
                "status": "error",
                "message": f"Workspace '{data['workspace']}' introuvable. Disponibles : {', '.join(names)}"
            }), 404

        session['api_info'] = data
        return jsonify({"status": "success", "message": "Connexion OK"})
    except ssaw.exceptions.UnauthorizedError:
        return jsonify({"status": "error", "message": "Identifiants incorrects"}), 401
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
# API POUR Initialiser une configuration



# API POUR RECUPERER LES STATISTIQUES DES INTERVIEWS
@app.route('/api/interview_stats', methods=['POST'])
def get_interview_stats_api():
    """
    Récupère les statistiques des interviews (nombre total, statuts, questionnaires) pour un workspace donné.
    """
    data = request.get_json()

    required_fields = ['api_url', 'username', 'password', 'workspace']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({
            "status": "error",
            "message": f"Champs manquants : {', '.join(missing_fields)}"
        }), 400

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=data['workspace']
        )

        interviews_api = InterviewsApi(client)
        interviews = list(interviews_api.get_list(fields=['id', 'status', 'questionnaire_id']))

        total_interviews = len(interviews)
        status_counts = defaultdict(int)
        questionnaire_ids = set()

        for interview in interviews:
            status_counts[interview.status] += 1
            questionnaire_ids.add(interview.questionnaire_id)

        return jsonify({
            "status": "success",
            "total_interviews": total_interviews,
            "status_counts": dict(status_counts),
            "questionnaire_count": len(questionnaire_ids)
        })

    except ssaw.exceptions.UnauthorizedError:
        return jsonify({
            "status": "error",
            "message": "Authentification échouée."
        }), 401
    except ssaw.exceptions.ForbiddenError:
        return jsonify({
            "status": "error",
            "message": "Accès interdit au workspace."
        }), 403
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
# API POUR RECUPERER LES STATISTIQUES DES INTERVIEWS



# API POUR RECUPERER LES TYPES D'EXPORTATION
@app.route('/api/export_types', methods=['GET'])
def get_export_types():
    """
    Retourne la liste des formats d'exportation disponibles dans Survey Solutions.
    """
    export_types = [
        "Tabular",         # CSV compressé (.zip)
        "STATA",           # .dta compressé
        "SPSS",            # .sav compressé
        "Binary",          # Format binaire interne
        "DDI",             # Data Documentation Initiative XML
        "Paradata"         # Données de log (paradata)
    ]

    return jsonify({
        "status": "success",
        "export_types": export_types,
        "count": len(export_types)
    })
# API POUR RECUPERER LES TYPES D'EXPORTATION


@app.route('/api/questionnaires', methods=['POST'])
def get_questionnaires_by_workspace():
    data = request.get_json()

    required_fields = ['api_url', 'username', 'password', 'workspace']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({
            "status": "error",
            "message": f"Champs manquants: {', '.join(missing)}"
        }), 400

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=data['workspace']
        )

        q_api = QuestionnairesApi(client)
        questionnaires = q_api.get_list()

        # Groupe fusionné : clé = (title, questionnaire_id)
        grouped = {}

        for q in questionnaires:
            key = (q.title, q.questionnaire_id)

            if key not in grouped:
                grouped[key] = {
                    "questionnaire_id": q.questionnaire_id,
                    "title": q.title,
                    "variable": q.variable or "Non défini",
                    "is_active": getattr(q, 'is_active', True),
                    "created_date": getattr(q, 'created_date', 'Non défini'),
                    "versions": [q.version]
                }
            else:
                grouped[key]["versions"].append(q.version)

        # Trie les versions décroissantes (facultatif)
        for item in grouped.values():
            item["versions"].sort(reverse=True)

        result = list(grouped.values())

        # Trier le résultat par titre
        result.sort(key=lambda x: x['title'])

        return jsonify({
            "status": "success",
            "count": len(result),
            "data": result
        })

    except ssaw.exceptions.UnauthorizedError:
        return jsonify({
            "status": "error",
            "message": "Authentification échouée, vérifiez vos identifiants."
        }), 401
    except ssaw.exceptions.ForbiddenError:
        return jsonify({
            "status": "error",
            "message": "Accès interdit au workspace. Vérifiez les droits."
        }), 403
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
# API POUR RECUPERER LES QUESTIONNAIRES D'UN WORKSPACE



# API POUR RECUPERER UN LES INFOS D'UN QUESTIONNAIRE EN PARTICULIER
@app.route('/api/select_questionnaire', methods=['POST'])
def api_select_questionnaire():
    data = request.get_json()
    questionnaire_id = data.get('questionnaire_id')

    if not questionnaire_id:
        return jsonify({"status": "error", "message": "Aucun questionnaire_id fourni."}), 400

    questionnaires = session.get('available_questionnaires', [])

    # Debug print
    print("Questionnaires en session:", questionnaires)

    # Recherche stricte sur la clé 'questionnaire_id'
    selected_questionnaire = next(
        (q for q in questionnaires if q.get('questionnaire_id') == questionnaire_id),
        None
    )

    if selected_questionnaire:
        session['selected_questionnaire'] = selected_questionnaire
        session.modified = True
        return jsonify({
            "status": "success",
            "message": f"Questionnaire sélectionné : {selected_questionnaire['title']} (v{selected_questionnaire['version']})",
            "selected": selected_questionnaire
        })
    else:
        return jsonify({"status": "error", "message": "Questionnaire non trouvé."}), 404
# API POUR RECUPERER UN LES INFOS D'UN QUESTIONNAIRE EN PARTICULIER



@app.route('/api/download', methods=['POST'])
def api_download_data():
    """
    API pour lancer un export Survey Solutions et télécharger le fichier,
    avec prise en compte de la version spécifique ou de la dernière version disponible.
    """
    data = request.get_json()

    required_fields = [
        'api_url', 'username', 'password', 'workspace', 
        'variable_qx', 'export_type', 'interview_status'
    ]
    for field in required_fields:
        if field not in data:
            return jsonify({"status": "error", "message": f"Champ requis manquant : {field}"}), 400

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=data['workspace']
        )

        export_type = data['export_type']
        interview_status = data['interview_status']
        valid_export_types = ['Tabular', 'STATA', 'SPSS', 'Binary', 'DDI', 'Paradata']
        valid_interview_statuses = [
            'All', 'SupervisorAssigned', 'InterviewerAssigned', 'Completed',
            'RejectedBySupervisor', 'ApprovedBySupervisor', 'RejectedByHeadquarters', 'ApprovedByHeadquarters'
        ]
        if export_type not in valid_export_types:
            return jsonify({"status": "error", "message": f"Format d'exportation invalide : {export_type}"}), 400
        if interview_status not in valid_interview_statuses:
            return jsonify({"status": "error", "message": f"Statut d'interviews invalide : {interview_status}"}), 400

        # Récupérer questionnaire
        variable_qx = data['variable_qx']
        q_api = QuestionnairesApi(client)
        all_versions = [q for q in q_api.get_list() if q.variable and q.variable.strip().lower() == variable_qx.lower()]

        if not all_versions:
            return jsonify({"status": "error", "message": f"Aucun questionnaire trouvé pour variable '{variable_qx}'"}), 404

        # Gestion version
        requested_version = data.get("version")
        if requested_version:
            matching = [q for q in all_versions if q.version == requested_version]
            if not matching:
                return jsonify({"status": "error", 
                    "message": f"Aucune version {requested_version} pour questionnaire '{variable_qx}'"}), 404
            q = matching[0]
        else:
            # Prendre la version la plus récente
            q = max(all_versions, key=lambda x: x.version)

        questionnaire_identity = f"{q.questionnaire_id}${q.version}"
        uuid.UUID(q.questionnaire_id)  # pour valider

        logging.info(f"Export demandé pour {q.title} (ID: {questionnaire_identity}, Format: {export_type}, Statut: {interview_status})")

        export_job = ExportJob(
            QuestionnaireId=questionnaire_identity,
            ExportType=export_type,
            InterviewStatus=interview_status
        )
        export_api = ExportApi(client)
        job = export_api.start(export_job=export_job, wait=True, show_progress=True)

        logging.info(f"Job créé : ID={job.job_id}, has_file={job.has_export_file}")

        if not (job and job.has_export_file and job.links and job.links.download):
            return jsonify({
                "status": "error",
                "message": f"Aucun fichier exporté généré. Statut: {getattr(job, 'export_status', 'NA')}"
            }), 404

        # Télécharger le fichier
        download_url = job.links.download
        auth_string = f"{data['username']}:{data['password']}"
        headers = {'Authorization': f'Basic {base64.b64encode(auth_string.encode()).decode()}'}
        response = requests.get(download_url, headers=headers, stream=True)

        if response.status_code != 200:
            return jsonify({"status": "error", "message": f"Téléchargement échoué. HTTP {response.status_code}"}), 500

        extension_map = {
            'STATA': '.dta.zip', 'SPSS': '.sav.zip', 'Tabular': '.csv.zip',
            'Binary': '.zip', 'DDI': '.xml.zip', 'Paradata': '.zip'
        }
        safe_title = re.sub(r'[^\w\-_\.]', '_', q.title or 'questionnaire')
        download_name = f"{safe_title}_v{q.version}_{export_type}_{interview_status}{extension_map.get(export_type, '.zip')}"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file_path = temp_file.name
            logging.info(f"Fichier export téléchargé vers {temp_file_path}")

        @after_this_request
        def cleanup(response):
            try:
                os.remove(temp_file_path)
                logging.info(f"Fichier temporaire supprimé : {temp_file_path}")
            except Exception as e:
                logging.error(f"Erreur suppression fichier temporaire : {e}")
            return response

        return send_file(temp_file_path, as_attachment=True,
                         download_name=download_name, mimetype='application/zip')

    except Exception as e:
        logging.error(f"Erreur lors du téléchargement : {e}")
        return jsonify({"status": "error", "message": f"Erreur interne : {str(e)}"}), 500


# API POUR ANNULER UNE EXPORTATION
@app.route('/api/cancel_export', methods=['POST'])
def api_cancel_export():
    """Annule un job d'exportation via une requête POST avec les infos en JSON (compatible Postman)."""
    data = request.json
    required_fields = ['api_url', 'username', 'password', 'workspace', 'export_job_id']

    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'message': 'Champs manquants dans le corps de la requête.'
        }), 400

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=data['workspace']
        )
        export_api = ExportApi(client)
        export_api.cancel(data['export_job_id'])
        logging.info(f"Job d'exportation {data['export_job_id']} annulé avec succès")

        return jsonify({'success': True, 'message': 'Job d\'exportation annulé avec succès.'})
    except ssaw.exceptions.UnauthorizedError:
        logging.error("Erreur d'authentification lors de l'annulation du job")
        return jsonify({'success': False, 'message': 'Identifiants API invalides.'}), 401
    except ssaw.exceptions.ForbiddenError:
        logging.error("Erreur de permission lors de l'annulation du job")
        return jsonify({'success': False, 'message': 'Accès interdit au workspace.'}), 403
    except Exception as e:
        logging.error(f"Erreur lors de l'annulation du job {data['export_job_id']} : {str(e)}")
        return jsonify({'success': False, 'message': f"Erreur lors de l'annulation : {str(e)}"}), 500
# API POUR ANNULER UNE EXPORTATION




# API POUR RECUPERER LES INTERVIEW PAR QUESTIONNAIRE
@app.route('/api/interviews_by_questionnaire', methods=['POST'])
def get_interviews_by_questionnaire():
    """Retourne les interviews d'un questionnaire spécifique (avec version)."""
    data = request.get_json()

    required_fields = ['api_url', 'username', 'password', 'workspace', 'questionnaire_guid', 'version']
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "Tous les champs sont requis, y compris la version."})

    try:
        client = Client(
            url=data['api_url'],
            api_user=data['username'],
            api_password=data['password'],
            workspace=data['workspace']
        )

        questionnaire_id = data['questionnaire_guid'].replace("-", "")
        version = int(data['version'])

        interviews_api = InterviewsApi(client)
        interviews_generator = interviews_api.get_list(
            questionnaire_identity={
                "questionnaireId": questionnaire_id,
                "version": version
            },
            fields=["id", "responsible", "questionnaireId", "status"]
        )

        interviews = list(interviews_generator)  # <== CONVERTIR en liste ici

        return jsonify({
            "status": "success",
            "message": f"{len(interviews)} interviews trouvées pour le questionnaire {data['questionnaire_guid']} v{version}.",
            "interviews": interviews
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erreur lors de la récupération des interviews : {str(e)}"
        })
# API POUR RECUPERER LES INTERVIEW PAR QUESTIONNAIRE



# API POUR REJETER ET COMMENTER AUTOMATIQUEMENT
def readable_to_server(readable_id: str, mapping: List[str]) -> Optional[str]:
    """Convertit un ID lisible en ID serveur si présent dans la liste de correspondance."""
    return readable_id if readable_id in mapping else None


def add_comments_and_reject_from_excel(client: Client, df: pd.DataFrame, mapping: List[str]):
    stats = {'total': 0, 'commented': 0, 'rejected': 0, 'errors': 0}
    status_breakdown = defaultdict(int)
    variables_stats = defaultdict(int)
    actions = []
    details = []

    interviews_to_reject = set()

    required_columns = ['interview__id', 'variable', 'comment']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        details.append(f"Colonnes manquantes dans le fichier Excel : {', '.join(missing)}")
        stats['errors'] = 1
        return stats, status_breakdown, variables_stats, actions, details

    stats['total'] = len(df)
    interviews_api = InterviewsApi(client)

    for _, row in df.iterrows():
        readable_id = str(row['interview__id'])
        variable = str(row['variable'])
        comment = str(row['comment'])
        roster_vector = json.loads(row.get('roster_vector', '[]')) if 'roster_vector' in df.columns else []

        action_record = {
            "interview_id": readable_id,
            "variable": variable,
            "comment_added": False,
            "rejected": False,
            "already_rejected": False
        }

        try:
            server_id = readable_to_server(readable_id, mapping)
            if not server_id:
                stats['errors'] += 1
                details.append(f"ID non trouvé dans la liste de correspondance : {readable_id}")
                continue

            interviews_api.comment(
                interview_id=server_id,
                variable=variable,
                comment=comment,
                roster_vector=roster_vector
            )
            stats['commented'] += 1
            variables_stats[variable] += 1
            action_record["comment_added"] = True
            details.append(f"Commentaire ajouté pour l'ID {server_id}, variable {variable}")

            interviews_to_reject.add(server_id)

        except Exception as e:
            stats['errors'] += 1
            details.append(f"Erreur lors du commentaire de l'ID {readable_id}, variable {variable} : {str(e)}")

        actions.append(action_record)

    for server_id in interviews_to_reject:
        try:
            interviews_api.reject(server_id, 'Rejeté après commentaire')
            stats['rejected'] += 1
            details.append(f"Rejeté : ID serveur ({server_id})")
            for action in actions:
                if action["interview_id"] == server_id:
                    action["rejected"] = True
        except Exception:
            stats['errors'] += 1
            details.append(f"Impossible de rejeter l'interview {server_id}: Il est déjà rejeté!")
            for action in actions:
                if action["interview_id"] == server_id:
                    action["already_rejected"] = True

    # Construire un breakdown final
    for action in actions:
        if action["already_rejected"]:
            status_breakdown["AlreadyRejected"] += 1
        elif action["rejected"]:
            status_breakdown["RejectedNow"] += 1
        elif action["comment_added"]:
            status_breakdown["CommentedOnly"] += 1

    return stats, dict(status_breakdown), dict(variables_stats), actions, details
# API POUR REJETER ET COMMENTER AUTOMATIQUEMENT


@app.route('/api/reject_and_comment', methods=['POST'])
def api_reject_and_comment():
    if 'excel_file' not in request.files:
        return jsonify({"error": "Fichier Excel manquant"}), 400

    file = request.files['excel_file']
    if not file.filename.endswith('.xlsx'):
        return jsonify({"error": "Le fichier doit être au format .xlsx"}), 400

    try:
        # Authentification
        api_user = request.form.get('api_url')
        utilisateur = request.form.get('username')
        mot_de_passe = request.form.get('password')
        workspace = request.form.get('workspace')

        if not all([api_user, utilisateur, mot_de_passe, workspace]):
            return jsonify({"error": "Paramètres d'authentification manquants"}), 400

        df = pd.read_excel(file)

        client = Client(
            url=api_user,
            api_user=utilisateur,
            api_password=mot_de_passe,
            workspace=workspace
        )

        # Liste des interviews dispos
        interviews_api = InterviewsApi(client)
        id_mapping = [interview.id for interview in interviews_api.get_list(fields=['id'])]

        # Exécution du traitement enrichi
        stats, status_breakdown, variables_stats, actions, details = add_comments_and_reject_from_excel(
            client, df, id_mapping
        )

        return jsonify({
            "status": "success",
            "message": "Traitement terminé",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "executed_by": utilisateur,
            "workspace": workspace,
            "stats": stats,
            "status_breakdown": status_breakdown,
            "variables_stats": variables_stats,
            "actions": actions,
            "details": details
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# API POUR REJETER ET COMMENTER AUTOMATIQUEMENT



if __name__ == '__main__':
    print("Démarrage de l'application Flask sur le port 5003...")
    app.run(debug=True, port=5003)