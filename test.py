if __name__ == "__main__":

    anomalies = detect_anomalies()

    print("\nHealthcare Capability Anomalies\n")

    for a in anomalies[:10]:
        print(a["facility"], "→", a["issue"])