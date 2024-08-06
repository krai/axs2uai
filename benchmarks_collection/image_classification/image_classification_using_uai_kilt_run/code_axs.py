def generate_fanspeed_command(device_to_fanspeed):
    if device_to_fanspeed == "":
        return ""

    dictionary = dict(item.split(":") for item in device_to_fanspeed.split(";") if ":" in item)
    parts = [f"speedai smi --set-fanspeed {value} -b {key}" for key, value in dictionary.items()]
    command = " && ".join(parts)
    command += " && "
    return command