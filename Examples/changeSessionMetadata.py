"""
---------------------------------------------------------------------------
OpenCap: changeSessionMetadata.py
---------------------------------------------------------------------------

Copyright 2022 Stanford University and the Authors

Author(s): Scott Uhlrich, Antoine Falisse

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This script allows you to change the metadata of the session. E.g., to change
the pose estimator used when reprocessing data in the cloud. This is mostly for
developer use.

The available options for metadata are:
    - scalingsetup:     upright_standing_pose
                        any_pose
    - openSimModel:     LaiUhlrich2022
                        LaiUhlrich2022_shoulder
    - posemodel:        openpose
                        hrnet
    - augmentermodel:   v0.2
                        v0.3
    - filterfrequency:  default
                        float number
    - datasharing:      Share processed data and identified videos
                        Share processed data and de-identified videos
                        Share processed data
                        Share no data


"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from utils import changeSessionMetadata
from utils import download_file
from utils import getDataDirectory
from utils import getNeutralTrialID
from utils import getTrialJson
from utils import importMetadata
from utils import makeRequestWithRetry
from utilsAPI import getAPIURL
from utilsAuth import getToken


API_URL = getAPIURL()
API_TOKEN = getToken()


def updateMetadataDict(metadata, newMetadata):
    addedKeys = set()
    for key in list(metadata.keys()):
        if key in newMetadata:
            metadata[key] = newMetadata[key]
            addedKeys.add(key)
        if isinstance(metadata[key], dict):
            for key2 in list(metadata[key].keys()):
                if key2 in newMetadata:
                    metadata[key][key2] = newMetadata[key2]
                    addedKeys.add(key2)

    for key, value in newMetadata.items():
        if key not in addedKeys:
            metadata[key] = value

    return metadata


def ensureLocalMetadata(session_id, session_path):
    metadata_path = os.path.join(session_path, 'sessionMetadata.yaml')
    if os.path.exists(metadata_path):
        return metadata_path

    os.makedirs(session_path, exist_ok=True)
    neutral_id = getNeutralTrialID(session_id)
    trial = getTrialJson(neutral_id)
    for result in trial['results']:
        if result['tag'] == 'session_metadata':
            download_file(result['media'], metadata_path)
            return metadata_path

    raise FileNotFoundError(
        'Could not find local metadata or session_metadata result for '
        '{}.'.format(session_id))


def changeLocalSessionMetadata(session_id, newMetadata):
    session_path = os.path.join(getDataDirectory(isDocker=False), 'Data',
                                session_id)
    metadata_path = ensureLocalMetadata(session_id, session_path)
    metadata = updateMetadataDict(importMetadata(metadata_path), newMetadata)

    with open(metadata_path, 'w') as file:
        yaml.dump(metadata, file)

    override_path = os.path.join(session_path, 'sessionMetadata_local.yaml')
    with open(override_path, 'w') as file:
        yaml.dump(newMetadata, file)

    print('Updated local metadata: {}'.format(metadata_path))
    print('Wrote local metadata override: {}'.format(override_path))


def hasWritePermissions(session_id):
    response = makeRequestWithRetry(
        'GET',
        API_URL + "sessions/{}/get_session_permission/".format(session_id),
        headers={"Authorization": "Token {}".format(API_TOKEN)})
    permissions = response.json()
    return permissions['isAdmin'] or permissions['isOwner']

session_ids = ["3375ffbc-daeb-4a43-b4f7-ac9899cd4c71"]

# Dictionary of metadata fields to change (see sessionMetadata.yaml).
newMetadata = {
    'openSimModel':'LaiUhlrich2022_adjusted',
}

for session_id in session_ids:
    if hasWritePermissions(session_id):
        changeSessionMetadata([session_id], newMetadata.copy())
    else:
        print('No write permission for {}; editing local metadata only.'.format(
            session_id))
        changeLocalSessionMetadata(session_id, newMetadata.copy())
