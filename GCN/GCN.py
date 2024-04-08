import networkx as nx
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_path = './GCN_model.pth'


def makeGraph(path):
    # Create an directed graph
    G = nx.DiGraph()
    with open(path) as fp:
        for line in fp:
            paper1, paper2 = line.split()
            G.add_edge(paper2, paper1)
            # print(paper1, "<---", paper2)
            
    return G


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def get_features():
    with open('../dataset/cora.content') as fp:
        features = {}
        for line in fp:
            parts = line.split()
            features[parts[0]] = {}
            features[parts[0]]['description'] = parts[1:-1]
            features[parts[0]]['label'] = parts[-1] 
    return features

def create_pyg_data_object(G, features):
    # Convert node labels to contiguous integers
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    edge_index = torch.tensor(
        [[node_mapping[src], node_mapping[dest]] for src, dest in G.edges()],
        dtype=torch.long
    ).t().contiguous()
    
    # Convert features to tensors directly
    x = torch.tensor(
        [[int(val) for val in features[node]['description']] for node in G.nodes()],
        dtype=torch.float
    )
    
    # Efficient label processing
    unique_labels = set(features[node]['label'] for node in G.nodes())
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y = torch.tensor(
        [label_mapping[features[node]['label']] for node in G.nodes()],
        dtype=torch.long
    )

    num_classes = len(unique_labels)
    data = Data(x=x, edge_index=edge_index, y=y, num_classes=num_classes)
    return data

def prepare_data_for_test(G_test, features):
    node_features_list = []
    node_labels_list = []
    for node in G_test.nodes():
        temp = [int(val) for val in features[node]['description']]
        node_features_list.append(temp)
        
    # Convert labels to integers
    labels = list(set(features[node]['label'] for node in G_test.nodes()))
    label_mapping = {label: i for i, label in enumerate(labels)}
    y_test = torch.tensor([label_mapping[features[node]['label']] for node in G_test.nodes()], dtype=torch.long)
    
    x_test = torch.tensor(node_features_list, dtype=torch.float)
    
    # Convert the graph to PyG format
    data_test = from_networkx(G_test)
    data_test.x = x_test
    data_test.y = y_test
    
    return data_test


def train_model(data):
    model = GCN(num_features=data.num_features, num_classes=data.num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):  # Adjust the number of epochs according to your needs
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:  # Change the logging frequency as needed
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return model


def evaluate_model(model, data_test):
    model.eval()
    model = model.to(device)
    data_test = data_test.to(device)
    with torch.no_grad():
        logits = model(data_test)
        preds = logits.argmax(dim=1)
        
        test_correct = preds.eq(data_test.y).sum().item()
        test_acc = test_correct / len(data_test.y)

        # Convert tensors to CPU for sklearn compatibility
        y_true = data_test.y.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Generate the classification report
        class_report = classification_report(y_true, y_pred, zero_division=0)

    print(f'Test Accuracy: {test_acc:.4f}')
    print("Classification Report:")
    print(class_report)
    
    output = f'Test Accuracy: {test_acc:.4f}\n\n' + "Classification Report:\n" + class_report
    

     # Write to file
    with open("./gcn_metrics.txt", "w") as f:
        f.write(output)


def save_model(model):
    torch.save(model.state_dict(), model_path)





def main():
    features = get_features()
    G_train =  makeGraph(path = '../dataset/cora_train.cites')
    data_train = create_pyg_data_object(G_train, features)
    print("training model")
    model = train_model(data_train)
    print("saving trained model for future use")
    save_model(model)
    G_test = makeGraph(path='../dataset/cora_test.cites')
    data_test = prepare_data_for_test(G_test, features)
    print("Evaluating model")
    evaluate_model(model, data_test)    
    
    
    
if __name__ == "__main__":
     main()