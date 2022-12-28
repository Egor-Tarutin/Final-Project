def save_mask(mask, file_path):
    with open(file_path, mode='w') as file:
        file.write(mask.__str__())