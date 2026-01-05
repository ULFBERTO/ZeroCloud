import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

interface SSMModelInfo {
  name: string;
  repoId: string;
  architecture: string;
  params: string;
  complexity: string;
  description: string;
}

@Component({
  selector: 'app-ssm-chat',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="min-h-screen bg-gray-900 text-white flex flex-col">
      <!-- Header -->
      <header class="bg-gray-800 border-b border-gray-700 p-4">
        <div class="max-w-4xl mx-auto flex items-center justify-between">
          <div class="flex items-center gap-3">
            <button
              (click)="goBack()"
              class="p-2 hover:bg-gray-700 rounded-lg transition-colors">
              ← Volver
            </button>
            <div>
              <h1 class="text-xl font-bold">🦀 {{ model.name }}</h1>
              <p class="text-sm text-gray-400">{{ model.architecture }}</p>
            </div>
          </div>
          
          <a 
            [href]="'https://huggingface.co/' + model.repoId"
            target="_blank"
            class="px-4 py-2 bg-yellow-600 hover:bg-yellow-500 rounded-lg text-sm transition-colors flex items-center gap-2">
            🤗 Ver en HuggingFace
          </a>
        </div>
      </header>

      <!-- Main Content -->
      <main class="flex-1 max-w-4xl mx-auto w-full p-6">
        <!-- Model Info Card -->
        <div class="bg-gray-800 rounded-xl p-6 mb-6">
          <div class="flex items-start gap-6">
            <div class="text-6xl">🧠</div>
            <div class="flex-1">
              <h2 class="text-2xl font-bold mb-2">{{ model.name }}</h2>
              <p class="text-gray-400 mb-4">{{ model.description }}</p>
              
              <div class="grid grid-cols-3 gap-4 text-sm">
                <div class="bg-gray-700/50 rounded-lg p-3">
                  <p class="text-gray-400">Arquitectura</p>
                  <p class="font-medium">{{ model.architecture }}</p>
                </div>
                <div class="bg-gray-700/50 rounded-lg p-3">
                  <p class="text-gray-400">Parámetros</p>
                  <p class="font-medium">{{ model.params }}</p>
                </div>
                <div class="bg-gray-700/50 rounded-lg p-3">
                  <p class="text-gray-400">Complejidad</p>
                  <p class="font-medium text-green-400">{{ model.complexity }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- SSM vs Transformer -->
        <div class="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl p-6 mb-6 border border-purple-500/30">
          <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
            ⚡ ¿Por qué SSM en lugar de Transformer?
          </h3>
          <div class="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <p class="text-gray-400 mb-2">Transformer (GPT, Llama, etc.)</p>
              <ul class="space-y-1 text-gray-300">
                <li>• Complejidad O(n²) cuadrática</li>
                <li>• Memoria crece exponencialmente</li>
                <li>• Contexto limitado (costoso)</li>
              </ul>
            </div>
            <div>
              <p class="text-purple-400 mb-2">SSM (Mamba-like) ✓</p>
              <ul class="space-y-1 text-gray-300">
                <li>• Complejidad O(n) lineal</li>
                <li>• Memoria crece linealmente</li>
                <li>• Contexto teóricamente ilimitado</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- How to Use -->
        <div class="bg-gray-800 rounded-xl p-6 mb-6">
          <h3 class="text-lg font-semibold mb-4">🚀 Cómo usar este modelo</h3>
          
          <div class="space-y-4">
            <!-- Step 1 -->
            <div class="flex gap-4">
              <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-bold shrink-0">1</div>
              <div class="flex-1">
                <p class="font-medium mb-2">Clonar el repositorio</p>
                <div class="bg-gray-900 rounded-lg p-3 font-mono text-sm overflow-x-auto">
                  <code>git clone https://huggingface.co/{{ model.repoId }}</code>
                </div>
              </div>
            </div>

            <!-- Step 2 -->
            <div class="flex gap-4">
              <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-bold shrink-0">2</div>
              <div class="flex-1">
                <p class="font-medium mb-2">Instalar dependencias</p>
                <div class="bg-gray-900 rounded-lg p-3 font-mono text-sm overflow-x-auto">
                  <code>pip install torch</code>
                </div>
              </div>
            </div>

            <!-- Step 3 -->
            <div class="flex gap-4">
              <div class="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-bold shrink-0">3</div>
              <div class="flex-1">
                <p class="font-medium mb-2">Ejecutar el chat</p>
                <div class="bg-gray-900 rounded-lg p-3 font-mono text-sm overflow-x-auto">
                  <code>python chat.py</code>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Python Code Example -->
        <div class="bg-gray-800 rounded-xl p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold">📝 Código de ejemplo</h3>
            <button 
              (click)="copyCode()"
              class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors">
              {{ copied() ? '✓ Copiado' : '📋 Copiar' }}
            </button>
          </div>
          
          <pre class="bg-gray-900 rounded-lg p-4 overflow-x-auto text-sm"><code class="text-gray-300">{{ pythonCode }}</code></pre>
        </div>
      </main>

      <!-- Footer -->
      <footer class="bg-gray-800 border-t border-gray-700 p-4 text-center text-sm text-gray-500">
        🦀 OxideLLM - Modelos de lenguaje experimentales con arquitecturas de vanguardia
      </footer>
    </div>
  `,
})
export class SSMChatComponent {
  private readonly router = inject(Router);
  
  readonly copied = signal(false);

  readonly model: SSMModelInfo = {
    name: 'OxideLLM_TK_SSM_V1',
    repoId: 'ULFBERTO/OxideLLM_TK_SSM_V1',
    architecture: 'SSM Selectivo (Mamba-like)',
    params: '~770K',
    complexity: 'O(n) Lineal',
    description: 'Transformer Killer - Modelo experimental que reemplaza la atención cuadrática de los Transformers con un State Space Model de complejidad lineal.',
  };

  readonly pythonCode = `import torch
from model import TransformerKiller
from tokenizer import CharacterTokenizer

# Cargar checkpoint
cp = torch.load("ssm_checkpoint.pth", map_location="cpu")

# Reconstruir tokenizer
tokenizer = CharacterTokenizer()
tokenizer.chars = cp['tokenizer_chars']
tokenizer.vocab_size = len(tokenizer.chars)
tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}

# Cargar modelo
model = TransformerKiller(
    vocab_size=tokenizer.vocab_size,
    dim=128,
    n_layers=4,
    state_dim=16
)
model.load_state_dict(cp['model_state_dict'])
model.eval()

# Generar texto
def generate(prompt, max_tokens=100, temperature=0.8):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(idx)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    return tokenizer.decode(idx[0].tolist())

print(generate("Hola, "))`;

  goBack(): void {
    this.router.navigate(['/']);
  }

  async copyCode(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.pythonCode);
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 2000);
    } catch (err) {
      console.error('Error copying:', err);
    }
  }
}
