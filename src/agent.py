import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class NormalizadorEstado:
    def __init__(self, dims_espacio_observacion):
        self.media = np.zeros(dims_espacio_observacion)
        self.std = np.ones(dims_espacio_observacion)
        self.contador = 0
        self.suma_acumulada = np.zeros(dims_espacio_observacion)
        self.suma_cuadrados_acumulada = np.zeros(dims_espacio_observacion)

    def normalizar(self, obs):
        if self.contador > 0:
            return (obs - self.media) / (self.std + 1e-8)
        return obs

    def actualizar(self, lote_obs):
        if lote_obs.ndim == 1:
            lote_obs = lote_obs[np.newaxis, :]

        self.suma_acumulada += np.sum(lote_obs, axis=0)
        self.suma_cuadrados_acumulada += np.sum(np.square(lote_obs), axis=0)
        self.contador += lote_obs.shape[0]

        if self.contador > 0:
            self.media = self.suma_acumulada / self.contador
            varianza = (self.suma_cuadrados_acumulada / self.contador) - np.square(self.media)
            self.std = np.sqrt(varianza + 1e-8)
            self.std[self.std < 1e-2] = 1e-2

class RedActorCritico(nn.Module):
    def __init__(self, dims_espacio_observacion, dims_espacio_accion):
        super(RedActorCritico, self).__init__()

        self.fc_comun1 = nn.Linear(dims_espacio_observacion, 256)
        self.relu_comun1 = nn.ReLU()
        self.fc_comun2 = nn.Linear(256, 256)
        self.relu_comun2 = nn.ReLU()

        self.capa_media_actor = nn.Linear(256, dims_espacio_accion)
        self.log_std = nn.Parameter(torch.full((dims_espacio_accion,), 0.0))

        self.capa_valor_critico = nn.Linear(256, 1)

    def forward(self, x):
        salida_comun = self.relu_comun1(self.fc_comun1(x))
        salida_comun = self.relu_comun2(self.fc_comun2(salida_comun))

        media = torch.tanh(self.capa_media_actor(salida_comun))
        std = torch.exp(self.log_std)
        std = torch.clamp(std, 0.1, 1.0)

        distribucion_accion = Normal(media, std)
        valor = self.capa_valor_critico(salida_comun)
        
        return distribucion_accion, valor

class AgentePPO:
    def __init__(self, dims_espacio_observacion, dims_espacio_accion,
                 parametro_clip=0.2, epocas_ppo=10, tamano_mini_lote=64,
                 lr_actor=3e-4, gamma=0.99, lambda_gae=0.95, coef_entropia=0.05):
        
        self.actor_critico = RedActorCritico(dims_espacio_observacion, dims_espacio_accion)
        self.optimizador = optim.Adam(self.actor_critico.parameters(), lr=lr_actor)

        self.parametro_clip = parametro_clip
        self.epocas_ppo = epocas_ppo
        self.tamano_mini_lote = tamano_mini_lote
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.coef_entropia = coef_entropia

        self.normalizador_estado = NormalizadorEstado(dims_espacio_observacion)

    def obtener_accion_y_valor(self, obs):
        self.normalizador_estado.actualizar(obs)
        obs_normalizada = self.normalizador_estado.normalizar(obs)
        tensor_obs = torch.FloatTensor(obs_normalizada).unsqueeze(0)
        
        dist_accion, valor = self.actor_critico(tensor_obs)
        accion = dist_accion.sample()
        log_prob = dist_accion.log_prob(accion).sum(dim=-1)
        
        return accion.squeeze(0).detach().numpy(), log_prob.item(), valor.item()

    def evaluar_accion_y_valor(self, tensor_obs, tensor_accion):
        dist_accion, valor = self.actor_critico(tensor_obs)
        log_prob = dist_accion.log_prob(tensor_accion).sum(dim=-1)
        entropia = dist_accion.entropy().sum(dim=-1)
        return log_prob, valor, entropia

    def calcular_gae(self, recompensas, terminados, valores):
        ventajas = []
        retornos = []
        ultima_gae_lambda = 0
        
        for t in reversed(range(len(recompensas))):
            if t == len(recompensas) - 1:
                siguiente_valor = 0
            else:
                siguiente_valor = valores[t+1] * (1 - terminados[t+1])

            delta = recompensas[t] + self.gamma * siguiente_valor * (1 - terminados[t]) - valores[t]
            ultima_gae_lambda = delta + self.gamma * self.lambda_gae * ultima_gae_lambda * (1 - terminados[t])
            
            ventajas.insert(0, ultima_gae_lambda)
            retornos.insert(0, ventajas[0] + valores[t])

        return np.array(retornos), np.array(ventajas)

    def actualizar(self, lote_obs, lote_acciones, lote_log_probs, lote_retornos, lote_ventajas):
        perdida_total_actor = 0
        perdida_total_critico = 0
        num_mini_lotes = 0

        for _ in range(self.epocas_ppo):
            indices = np.arange(len(lote_obs))
            np.random.shuffle(indices)
            
            for idx_inicio in range(0, len(lote_obs), self.tamano_mini_lote):
                idx_fin = idx_inicio + self.tamano_mini_lote
                indices_mini_lote = indices[idx_inicio:idx_fin]

                mini_obs = lote_obs[indices_mini_lote]
                mini_acciones = lote_acciones[indices_mini_lote]
                mini_log_probs_antiguos = lote_log_probs[indices_mini_lote]
                mini_retornos = lote_retornos[indices_mini_lote]
                mini_ventajas = lote_ventajas[indices_mini_lote]

                nuevos_log_probs, nuevos_valores, entropia = self.evaluar_accion_y_valor(mini_obs, mini_acciones)
                
                perdida_critico = (nuevos_valores.squeeze() - mini_retornos).pow(2).mean()

                ratio = torch.exp(nuevos_log_probs - mini_log_probs_antiguos)
                surr1 = ratio * mini_ventajas
                surr2 = torch.clamp(ratio, 1.0 - self.parametro_clip, 1.0 + self.parametro_clip) * mini_ventajas
                
                perdida_actor = -torch.min(surr1, surr2).mean() - self.coef_entropia * entropia.mean()

                self.optimizador.zero_grad()
                perdida_total = perdida_actor + perdida_critico
                perdida_total.backward()
                nn.utils.clip_grad_norm_(self.actor_critico.parameters(), 0.5)
                self.optimizador.step()

                perdida_total_actor += perdida_actor.item()
                perdida_total_critico += perdida_critico.item()
                num_mini_lotes += 1
        
        return perdida_total_actor / num_mini_lotes, perdida_total_critico / num_mini_lotes