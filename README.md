# Deep-Learning-model-to-detect-DDoS-attacks

### About
With the advancement of technology and the internet, network assaults take many different forms. Denial-of-Service (DoS) attacks are one of the most difficult security risks that the internet now faces. Attacks known as distributed denial of service (DDoS) are particularly worrisome since they can seriously impair a victim's computer or communication capabilities with little to no prior notice. Due to their complexity and frequency, these attacks are challenging to recognize and defend against. Wireshark can be used to find a spike in traffic during a DDoS assault.

Also used to identify DDoS flooding attacks are a number of machine learning techniques, including SGD, KNN, Logistic Regression, Multi-layer Perceptron, Support Vector Machine, Naive Bayes, XGboost, Quadratic discriminant, Decision Tree, and deep neural networks. Through the use of accuracy metrics, these algorithms are analyzed and contrasted.

### Dataset Description
The dataset used in this research is made up of SDN-specific data produced by the Mininet emulator, mainly for the purpose of differentiating network traffic. We put up 10 topologies in Mininet and connected switches to a single Ryu controller to create this dataset. We performed network simulations with benign TCP, ICMP, and UDP traffic as well as data collection for malicious TCP Syn, ICMP, and UDP flood attacks. The dataset includes 23 features, including switches- and metrics-derived features:
1. "packet_count" indicates the number of transmitted packets.
2. "byte_count" specifies the volume of sent data.
3. "Switch-id" serves as the unique switch identifier.
4. "duration_sec" and "duration_nsec" measure time in seconds and nanoseconds for packet transmission.
5. "Source IP" and "Destination IP" reveal the source and destination machine IP addresses, respectively.
6. "Port Number" identifies the port to which packets are directed.
7. "tx_bytes" and "rx_bytes" showcase data leaving and entering the switch port.
8. The "dt field" contains date and time converted into a numerical format.
9. Flow monitoring occurs at 30-second intervals.

### Proposed Methodology:

This dataset is designed specifically for Software-Defined Networks (SDN) and is intended for deep learning. It is produced by designing host-to-host Mininet topologies. Python programs gather flow and port information, combining stats into a single dataset. Preprocessing and model construction are aided by libraries like sklearn, TensorFlow, and Pandas. The dataset is prepared, encoded, and divided into training and testing groups. The application of baseline classifiers (DNN, KNN, SVM, etc.) results in the maximum accuracy (99.19%) being produced by DNN. Tuning the hyperparameters improves model performance. The resulting model avoids overfitting and displays a good level of accuracy with an AUC of 0.9998218809615622.

