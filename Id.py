def identify_fault(row):
    faults = []
    if row["OILT1__deg"].values[0] > 110:
        faults.append("Oil System Overheating")
    if row["EGT_1__deg"].values[0] > 850:
        faults.append("Exhaust Temperature High")
    if row["FUEP1__psi"].values[0] < 20:
        faults.append("Fuel Pressure Drop")
    if row["N1__1_pcnt"].values[0] < 50 and row["rpm_1engin"].values[0] < 1500:
        faults.append("Combustion Instability")
    
    if not faults:
        return "No faults detected — system nominal."
    else:
        return ", ".join(faults)
