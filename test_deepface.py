import numpy as np
from deepface import DeepFace

dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

for backend in ["ssd", "mtcnn", "retinaface"]:
    try:
        face_objs = DeepFace.extract_faces(
            img_path=dummy_img,
            detector_backend=backend,
            enforce_detection=False,
            align=False
        )
        print(f'{backend} WORKS!')
    except Exception as e:
        print(f'ERROR with {backend}: {e}')
