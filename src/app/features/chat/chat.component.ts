import {
  Component,
  inject,
  signal,
  computed,
  ViewChild,
  ElementRef,
  AfterViewChecked,
  OnInit,
  OnDestroy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import {
  ChatBackendInterface,
  ChatMessage,
} from '../../core/interfaces/chat-backend.interface';
import { WebLLMService, GPUInfo } from '../../core/services/webllm.service';
import { ModelManagerService } from '../../core/services/model-manager.service';
import { ChatHistoryService } from '../../core/services/chat-history.service';
import { P2PSyncService } from '../../core/services/p2p-sync.service';
import { GPUClusterService } from '../../core/services/gpu-cluster.service';
import { DistributedInferenceService } from '../../core/services/distributed-inference.service';
import { SyncModalComponent } from './sync-modal/sync-modal.component';
import { ClusterPanelComponent } from './cluster-panel/cluster-panel.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, SyncModalComponent, ClusterPanelComponent],
  providers: [
    { provide: ChatBackendInterface, useClass: WebLLMService },
  ],
  templateUrl: './chat.component.html',
})
export class ChatComponent implements AfterViewChecked, OnInit, OnDestroy {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;

  private readonly chatBackend = inject(ChatBackendInterface);
  private readonly webllmService = inject(WebLLMService);
  private readonly modelManager = inject(ModelManagerService);
  private readonly chatHistory = inject(ChatHistoryService);
  private readonly p2pSync = inject(P2PSyncService);
  private readonly gpuCluster = inject(GPUClusterService);
  private readonly distributedInference = inject(DistributedInferenceService);
  private readonly router = inject(Router);

  readonly selectedModel = this.modelManager.selectedModel;
  readonly sessions = this.chatHistory.sortedSessions;
  readonly activeSession = this.chatHistory.activeSession;

  // GPU selection
  readonly availableGPUs = this.webllmService.availableGPUs;
  readonly selectedGPU = this.webllmService.selectedGPU;
  readonly showGPUSelector = signal(false);

  readonly state = this.chatBackend.state;
  readonly currentResponse = this.chatBackend.currentResponse;
  readonly messages = signal<ChatMessage[]>([]);
  readonly userInput = signal<string>('');

  // UI state
  readonly showSidebar = signal(true);
  readonly editingMessageIndex = signal<number | null>(null);
  readonly editingContent = signal<string>('');
  readonly showSyncModal = signal(false);

  // P2P state
  readonly isP2PConnected = this.p2pSync.isConnected;
  readonly connectedPeersCount = computed(() => this.p2pSync.connectedPeers().length);

  // GPU Cluster state
  readonly showClusterPanel = signal(false);
  readonly isClusterActive = computed(() => this.gpuCluster.localNode() !== null);
  readonly canUseDistributed = this.distributedInference.canUseDistributed;

  private p2pTaskHandler = ((event: Event) => this.handleP2PTask(event as CustomEvent)) as EventListener;

  readonly canSend = computed(
    () =>
      this.state().isInitialized &&
      !this.state().isGenerating &&
      this.userInput().trim().length > 0
  );

  readonly displayMessages = computed(() => {
    const msgs = [...this.messages()];
    const streaming = this.currentResponse();
    if (this.state().isGenerating && streaming) {
      msgs.push({
        role: 'assistant',
        content: streaming,
        timestamp: new Date(),
      });
    }
    return msgs;
  });

  ngOnInit(): void {
    // Cargar sesión activa si existe
    const session = this.activeSession();
    if (session) {
      this.messages.set([...session.messages]);
    }

    // Detectar GPUs disponibles
    this.webllmService.detectAvailableGPUs();

    // Escuchar tareas P2P
    window.addEventListener('p2p-inference-task', this.p2pTaskHandler);
  }

  onGPUChange(gpuId: string): void {
    this.webllmService.selectGPU(gpuId);
    this.showGPUSelector.set(false);
  }

  toggleGPUSelector(): void {
    this.showGPUSelector.update(v => !v);
  }

  async clearCacheAndRetry(): Promise<void> {
    await this.webllmService.clearModelCache();
    await this.initializeModel();
  }

  ngOnDestroy(): void {
    window.removeEventListener('p2p-inference-task', this.p2pTaskHandler);
  }

  private async handleP2PTask(event: CustomEvent): Promise<void> {
    if (!this.state().isInitialized) return;

    const { taskId, prompt, fromId } = event.detail;

    try {
      const response = await this.chatBackend.sendMessage(prompt, []);
      this.p2pSync.sendInferenceResult(taskId, fromId, response);
    } catch (error) {
      console.error('P2P inference error:', error);
    }
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  async initializeModel(): Promise<void> {
    try {
      await this.chatBackend.initialize();
      // Crear nueva sesión si no hay una activa
      if (!this.activeSession()) {
        this.createNewChat();
      }
    } catch (error) {
      console.error('Error initializing model:', error);
    }
  }

  async sendMessage(): Promise<void> {
    const content = this.userInput().trim();
    if (!content || !this.canSend()) return;

    // Asegurar que hay una sesión activa
    if (!this.activeSession()) {
      this.createNewChat();
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    this.messages.update((msgs) => [...msgs, userMessage]);
    this.userInput.set('');

    try {
      const history = this.messages().filter((m) => m.role !== 'system');
      
      // Determinar modo de inferencia
      const isHost = this.p2pSync.connectionMode() === 'hosting';
      const hasPeers = this.p2pSync.connectedPeers().length > 0;
      const useDistributedGPU = this.canUseDistributed();
      
      let response: string;
      
      if (useDistributedGPU) {
        // Usar inferencia distribuida con WebGPU Compute Sharing
        response = await this.distributedInference.runWithFallback(content);
      } else if (isHost && hasPeers) {
        // Distribuir tarea a los peers (modo simple)
        response = await this.sendDistributedMessage(content, history.slice(0, -1));
      } else {
        // Procesar localmente
        response = await this.chatBackend.sendMessage(content, history.slice(0, -1));
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      this.messages.update((msgs) => [...msgs, assistantMessage]);

      // Guardar en historial
      const sessionId = this.activeSession()?.id;
      if (sessionId) {
        this.chatHistory.updateSessionMessages(sessionId, this.messages());
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  }

  private async sendDistributedMessage(content: string, history: ChatMessage[]): Promise<string> {
    return new Promise((resolve) => {
      // Enviar tarea a los peers
      const taskId = this.p2pSync.sendInferenceTask(content);
      
      // Escuchar resultado
      const resultHandler = (event: Event) => {
        const detail = (event as CustomEvent).detail;
        if (detail.taskId === taskId) {
          window.removeEventListener('p2p-inference-result', resultHandler);
          resolve(detail.result);
        }
      };
      
      window.addEventListener('p2p-inference-result', resultHandler);
      
      // Timeout de 60 segundos - si no hay respuesta, procesar localmente
      setTimeout(async () => {
        window.removeEventListener('p2p-inference-result', resultHandler);
        console.log('P2P timeout, processing locally...');
        const localResponse = await this.chatBackend.sendMessage(content, history);
        resolve(localResponse);
      }, 60000);
    });
  }

  stopGeneration(): void {
    this.chatBackend.abortGeneration();
  }

  clearChat(): void {
    this.messages.set([]);
    this.chatBackend.resetChat();
  }

  goToDashboard(): void {
    this.router.navigate(['/']);
  }

  openSyncModal(): void {
    this.showSyncModal.set(true);
  }

  closeSyncModal(): void {
    this.showSyncModal.set(false);
  }

  toggleClusterPanel(): void {
    this.showClusterPanel.update(v => !v);
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  // === Historial ===

  toggleSidebar(): void {
    this.showSidebar.update((v) => !v);
  }

  createNewChat(): void {
    const modelId = this.modelManager.selectedModelId();
    this.chatHistory.createSession(modelId);
    this.messages.set([]);
    this.chatBackend.resetChat();
  }

  loadSession(sessionId: string): void {
    this.chatHistory.setActiveSession(sessionId);
    const session = this.chatHistory.activeSession();
    if (session) {
      this.messages.set([...session.messages]);
      this.chatBackend.resetChat();
    }
  }

  deleteSession(sessionId: string, event: Event): void {
    event.stopPropagation();
    if (confirm('¿Eliminar esta conversación?')) {
      this.chatHistory.deleteSession(sessionId);
      if (this.activeSession()?.id === sessionId) {
        this.messages.set([]);
      }
    }
  }

  // === Edición de mensajes ===

  startEditMessage(index: number): void {
    const msg = this.messages()[index];
    if (msg.role !== 'user') return;
    this.editingMessageIndex.set(index);
    this.editingContent.set(msg.content);
  }

  cancelEdit(): void {
    this.editingMessageIndex.set(null);
    this.editingContent.set('');
  }

  async saveEditAndRegenerate(): Promise<void> {
    const index = this.editingMessageIndex();
    if (index === null) return;

    const sessionId = this.activeSession()?.id;
    if (!sessionId) return;

    const newContent = this.editingContent().trim();
    if (!newContent) return;

    // Editar mensaje y cortar historial
    const newMessages = this.chatHistory.editMessageAt(sessionId, index, newContent);
    this.messages.set(newMessages);
    this.cancelEdit();

    // Regenerar respuesta
    this.chatBackend.resetChat();
    try {
      const history = newMessages.filter((m) => m.role !== 'system');
      const response = await this.chatBackend.sendMessage(newContent, history.slice(0, -1));

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      this.messages.update((msgs) => [...msgs, assistantMessage]);
      this.chatHistory.updateSessionMessages(sessionId, this.messages());
    } catch (error) {
      console.error('Error regenerating:', error);
    }
  }

  // Fork desde un punto específico
  forkFromHere(index: number): void {
    const sessionId = this.activeSession()?.id;
    if (!sessionId) return;

    const newSession = this.chatHistory.forkFromMessage(sessionId, index);
    this.messages.set([...newSession.messages]);
    this.chatBackend.resetChat();
  }

  private scrollToBottom(): void {
    try {
      const el = this.messagesContainer?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    } catch {}
  }
}
