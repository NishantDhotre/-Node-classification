import networkx as nx
from gensim.models import Word2Vec
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def makeGraph(path):
    # Create an directed graph
    G = nx.DiGraph()
    with open(path) as fp:
        for line in fp:
            paper1, paper2 = line.split()
            G.add_edge(paper2, paper1)
            # print(paper1, "<---", paper2)
            
    return G

def get_features():
    with open('../dataset/cora.content') as fp:
        features = {}
        for line in fp:
            parts = line.split()
            features[parts[0]] = {}
            features[parts[0]]['description'] = parts[1:-1]
            features[parts[0]]['label'] = parts[-1] 
    return features

def node2vec_walk(graph, start_node, walk_length, p, q):
    """Perform a node2vec random walk from a start node."""
    walk = [start_node]
    while len(walk) < walk_length:
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        if len(walk) == 1:
            next_node = random.choice(neighbors)
        else:
            prev_node = walk[-2]
            probabilities = [1.0/q if neighbor == prev_node else 1.0/p if neighbor not in graph[prev_node] else 1.0 for neighbor in neighbors]
            probabilities_sum = sum(probabilities)
            probabilities = [prob/probabilities_sum for prob in probabilities]
            next_node = np.random.choice(neighbors, p=probabilities)
        walk.append(next_node)
    return walk



def generate_walks(graph, num_walks, walk_length, p=1, q=1):
    """Generate walks from the graph."""
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(node2vec_walk(graph, node, walk_length, p, q))
    return walks



def create_node2vec_embeddings(graph, dimensions=64, walk_length=30, num_walks=200, window_size=10, workers=4, p=1, q=1):
    """Generate node2vec embeddings."""
    walks = generate_walks(graph, num_walks, walk_length, p=p, q=q)
    walks = [list(map(str, walk)) for walk in walks]  # Convert node IDs to strings for gensim
    model = Word2Vec(sentences=walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=workers)
    embeddings = {node: model.wv[str(node)] for node in graph.nodes()}
    return embeddings



def pred_data(embeddings, features): 
    # Prepare the data
    X = np.array([embeddings[node] for node in embeddings])
    y = np.array([features[node]['label'] for node in embeddings])
    # Encode labels if they are not numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

def do_evaluation(y_test, y_pred):
    # Generate the classification report
    report = classification_report(y_test, y_pred, digits=4)
    confusion_mat = confusion_matrix(y_test, y_pred)

    output = "Classification Report:\n" + report + "\nConfusion Matrix:\n" + str(confusion_mat)

    # Print to console
    print(output)
    
    # Write to file
    with open("lr_metrics.txt", "w") as f:
        f.write(output)
        
        
def main():
    print("making graph..")
    G_train =  makeGraph(path = '../dataset/cora_train.cites')
    features = get_features()
    
    print("making embeddings..")
    train_embeddings = create_node2vec_embeddings(G_train)
    X_train, y_train = pred_data(train_embeddings, features)
    
    G_test = makeGraph(path='../dataset/cora_test.cites')
    test_embeddings = create_node2vec_embeddings(G_test)
    X_test, y_test = pred_data(test_embeddings, features)


    print("making logistic model..")
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)


    print("evaluating logistic model..")
    # Predict on the test set
    y_pred = model.predict(X_test)

    do_evaluation(y_test, y_pred)
   
   
    
if __name__ == "__main__":
     main()