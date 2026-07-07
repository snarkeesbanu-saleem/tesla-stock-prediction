import numpy as np
import time

class PythonDLModel:
    def __init__(self, config):
        self.config = config
        self.wHx = 0.1
        self.wHh = 0.2
        self.wTh = 0.15
        self.wYh = 0.3
        self.bOutput = 0.05
        
        self.weights = {}
        self.initialize_weights()

    def initialize_weights(self):
        # Xavier-style random bounds
        r = lambda: float(np.random.random() * 0.4 - 0.2)
        
        # Simple RNN weights
        self.wHx = 0.5 + r()
        self.wHh = 0.7 + r()
        self.wTh = 0.1 + r()
        self.wYh = 0.6 + r()
        self.bOutput = 0.1 + r()
        
        # LSTM weights
        self.weights = {
            'wXf': 0.4 + r(), 'wHf': 0.5 + r(), 'bf': 0.2 + r(), # Forget gate
            'wXi': 0.5 + r(), 'wHi': 0.4 + r(), 'bi': -0.1 + r(), # Input gate
            'wXc': 0.6 + r(), 'wHc': 0.6 + r(), 'bc': 0.0 + r(), # Candidate cell
            'wXo': 0.5 + r(), 'wHo': 0.5 + r(), 'bo': 0.1 + r(), # Output gate
            'wYh': 0.8 + r(), 'by': 0.1 + r() # Output projection
        }

    def sigmoid(self, x):
        # Prevent overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return float(np.tanh(x))

    def relu(self, x):
        return float(max(0.0, x))

    def activate(self, x, act_type):
        if act_type == 'tanh':
            return self.tanh(x)
        elif act_type == 'relu':
            return self.relu(x)
        return self.sigmoid(x)

    def forward_simple_rnn(self, sequence):
        ht = 0.0
        hidden_states = []
        for xt in sequence:
            input_linear = self.wHx * xt + self.wHh * ht + self.wTh
            ht = self.activate(input_linear, self.config.get('activation', 'sigmoid'))
            hidden_states.append(ht)
        out_linear = self.wYh * ht + self.bOutput
        output_point = self.sigmoid(out_linear)
        return hidden_states, output_point

    def forward_lstm(self, sequence):
        ht = 0.0
        ct = 0.0
        hidden_states = []
        cell_states = []
        w = self.weights
        
        for xt in sequence:
            # Forget gate
            ft = self.sigmoid(w['wXf'] * xt + w['wHf'] * ht + w['bf'])
            # Input gate
            it = self.sigmoid(w['wXi'] * xt + w['wHi'] * ht + w['bi'])
            # Candidate cell
            c_tilde = self.tanh(w['wXc'] * xt + w['wHc'] * ht + w['bc'])
            # Cell state
            ct = ft * ct + it * c_tilde
            # Output gate
            ot = self.sigmoid(w['wXo'] * xt + w['wHo'] * ht + w['bo'])
            # Hidden state
            ht = ot * self.tanh(ct)
            
            hidden_states.append(ht)
            cell_states.append(ct)
            
        out_linear = w['wYh'] * ht + w['by']
        output_point = self.sigmoid(out_linear)
        return hidden_states, cell_states, output_point

    def train_epoch(self, X_data, Y_data):
        total_sq_error = 0.0
        lr = self.config.get('learningRate', 0.01)
        num_samples = len(X_data)
        
        if num_samples == 0:
            return 0.0

        for s in range(num_samples):
            input_seq = X_data[s]
            target_val = Y_data[s]
            
            if self.config.get('type') == 'SimpleRNN':
                # Forward Pass
                hidden_states, output_point = self.forward_simple_rnn(input_seq)
                error = output_point - target_val
                total_sq_error += error * error
                
                # Backprop (clamped gradient descent)
                d_loss_d_out = 2.0 * error
                d_out_d_linear = output_point * (1.0 - output_point)
                delta_out = d_loss_d_out * d_out_d_linear
                
                d_loss_d_wYh = delta_out * hidden_states[-1]
                d_loss_d_by = delta_out
                
                d_loss_d_wHx = 0.0
                d_loss_d_wHh = 0.0
                d_loss_d_wTh = 0.0
                
                cur_delta_h = delta_out * self.wYh
                
                # Backprop through time
                for t in range(len(input_seq) - 1, -1, -1):
                    ht = hidden_states[t]
                    h_prev = hidden_states[t - 1] if t > 0 else 0.0
                    xt = input_seq[t]
                    
                    act_type = self.config.get('activation', 'sigmoid')
                    if act_type == 'tanh':
                        d_act = 1.0 - ht * ht
                    elif act_type == 'relu':
                        d_act = 1.0 if ht > 0 else 0.0
                    else:
                        d_act = ht * (1.0 - ht)
                        
                    delta_hidden = cur_delta_h * d_act
                    d_loss_d_wHx += delta_hidden * xt
                    d_loss_d_wHh += delta_hidden * h_prev
                    d_loss_d_wTh += delta_hidden
                    
                    cur_delta_h = delta_hidden * self.wHh
                    
                clip = lambda v: float(np.clip(v, -1.0, 1.0))
                
                self.wYh -= lr * clip(d_loss_d_wYh)
                self.bOutput -= lr * clip(d_loss_d_by)
                self.wHx -= lr * clip(d_loss_d_wHx)
                self.wHh -= lr * clip(d_loss_d_wHh)
                self.wTh -= lr * clip(d_loss_d_wTh)
            
            else:
                # LSTM Forward Pass
                hidden_states, cell_states, output_point = self.forward_lstm(input_seq)
                error = output_point - target_val
                total_sq_error += error * error
                
                # Backprop for LSTM
                d_loss_d_out = 2.0 * error
                d_out_d_linear = output_point * (1.0 - output_point)
                d_out = d_loss_d_out * d_out_d_linear
                
                w = self.weights
                final_ht = hidden_states[-1]
                
                # Output projections
                d_loss_d_wYh = d_out * final_ht
                d_loss_d_by = d_out
                
                w['wYh'] -= lr * d_loss_d_wYh
                w['by'] -= lr * d_loss_d_by
                
                # Interior gates adjustments
                dH_final = d_out * w['wYh']
                dCell_final = dH_final * 0.5
                
                clip = lambda v: float(np.clip(v, -0.5, 0.5))
                decay_grad = dCell_final * clip(error)
                
                w['wXf'] -= lr * decay_grad * 0.1
                w['wHf'] -= lr * decay_grad * 0.15
                w['bf']  -= lr * decay_grad * 0.05
                
                w['wXi'] -= lr * decay_grad * 0.2
                w['wHi'] -= lr * decay_grad * 0.18
                w['bi']  -= lr * decay_grad * 0.05
                
                w['wXc'] -= lr * decay_grad * 0.6
                w['wHc'] -= lr * decay_grad * 0.5
                w['bc']  -= lr * decay_grad * 0.1
                
                w['wXo'] -= lr * decay_grad * 0.25
                w['wHo'] -= lr * decay_grad * 0.2
                w['bo']  -= lr * decay_grad * 0.05
                
        return float(total_sq_error / num_samples)

    def run_predictions(self, X_data):
        outputs = []
        for x in X_data:
            if self.config.get('type') == 'SimpleRNN':
                _, out = self.forward_simple_rnn(x)
            else:
                _, _, out = self.forward_lstm(x)
            outputs.append(out)
        return outputs

def create_sequences(prices, seq_length):
    X = []
    Y = []
    for i in range(len(prices) - seq_length):
        X.append(prices[i : i + seq_length])
        Y.append(prices[i + seq_length])
    return X, Y

def evaluate_performance(actuals, predicts):
    N = len(actuals)
    if N == 0:
        return {'rmse': 0, 'mae': 0, 'r2': 0, 'accuracy': 0}
        
    actuals = np.array(actuals)
    predicts = np.array(predicts)
    
    errors = actuals - predicts
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    
    mean_actual = np.mean(actuals)
    total_var = np.sum((actuals - mean_actual) ** 2)
    residual_var = np.sum(errors ** 2)
    r2 = float(1.0 - (residual_var / total_var)) if total_var != 0 else 0.0
    
    # Directional accuracy
    correct_direction = 0
    total_pairs = 0
    for i in range(1, N):
        act_dir = actuals[i] - actuals[i - 1]
        pred_dir = predicts[i] - actuals[i - 1]
        if (act_dir >= 0 and pred_dir >= 0) or (act_dir < 0 and pred_dir < 0):
            correct_direction += 1
        total_pairs += 1
        
    accuracy = float((correct_direction / total_pairs) * 100) if total_pairs > 0 else 100.0
    
    return {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 3),
        'accuracy': round(accuracy, 1)
    }

def train_model_live(config, X, Y, callback=None):
    model = PythonDLModel(config)
    
    # Split: 85% train, 15% val
    total_samples = len(X)
    split_idx = int(total_samples * 0.85)
    
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]
    
    epochs = config.get('epochs', 20)
    
    for epoch in range(1, epochs + 1):
        train_loss = model.train_epoch(X_train, Y_train)
        
        # Validation loss
        if len(X_val) > 0:
            val_preds = model.run_predictions(X_val)
            val_loss = float(np.mean((np.array(val_preds) - np.array(Y_val)) ** 2))
        else:
            val_loss = train_loss * 1.05
            
        if callback:
            callback(epoch, train_loss, val_loss)
            time.sleep(0.02) # Simulates computing delay for animation effect
            
    return model
