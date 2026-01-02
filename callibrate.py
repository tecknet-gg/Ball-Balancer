def callibrate(servoOffsets):
    command = input("Enter command: ")
    if command.lower() == "exit":
        return False
    if len(command) < 3:
        print("Command too short")
        return servoOffsets
    try:
        servo = int(command[0]) - 1
    except ValueError:
        print("Invalid servo number")
        return servoOffsets
    if servo < 0 or servo > 2:
        print("Invalid servo")
        return servoOffsets
    op = command[1]
    if op not in ["+", "-", "="]:
        print("Invalid operator")
        return servoOffsets
    angleIn = command[2:].strip()
    try:
        angle = float(angleIn)
    except ValueError:
        print("Invalid angle")
        return servoOffsets
    if op == "+":
        servoOffsets[servo] += angle
    elif op == "-":
        servoOffsets[servo] -= angle
    else:
        servoOffsets[servo] = angle
    print("Updated servo offsets:", servoOffsets)

    return servoOffsets