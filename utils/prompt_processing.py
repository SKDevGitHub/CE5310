#CREATES A PROMPT FOR THE LANGUAGE MODEL
def create_prompt(detections, user_prompt):
    if not detections:
        return f"The image contains no detected objects. {user_prompt}"

    object_list = ", ".join([d["class"] for d in detections])
    return f"The image contains the following objects: {object_list}. {user_prompt}"