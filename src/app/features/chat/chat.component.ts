import {
  AfterViewChecked,
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  ElementRef,
  inject,
  OnDestroy,
  OnInit,
  signal,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { ChatBackendInterface, ChatMessage } from '../../core/interfaces/chat-backend.interface';
import { ChatHistoryService, ChatSession } from '../../core/services/chat-history.service';
import { DistributedInferenceService } from '../../core/services/distributed-inference.service';
import { GPUClusterService } from '../../core/services/gpu-cluster.service';
import { ModelManagerService } from '../../core/services/model-manager.service';
import { P2PSyncService } from '../../core/services/p2p-sync.service';
import { SSMWebGPUService } from '../../core/services/ssm-webgpu.service';
import { WebLLMService } from '../../core/services/webllm.service';
import { copyText } from '../../core/utils/clipboard';
import { formatRelativeTime } from '../../core/utils/format';
import {
  BadgeComponent,
  ButtonComponent,
  DialogComponent,
  IconComponent,
  SheetComponent,
} from '../../shared/components';
import { ClusterPanelComponent } from './cluster-panel/cluster-panel.component';
import { SyncModalComponent } from './sync-modal/sync-modal.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    FormsModule,
    RouterLink,
    SyncModalComponent,
    ClusterPanelComponent,
    SheetComponent,
    ButtonComponent,
    BadgeComponent,
    IconComponent,
    DialogComponent,
  ],
  providers: [{ provide: ChatBackendInterface, useClass: WebLLMService }],
  templateUrl: './chat.component.html',
})
export class ChatComponent implements AfterViewChecked, OnInit, OnDestroy {
  @ViewChild('messagesContainer') private messagesContainer?: ElementRef<HTMLElement>;
  @ViewChild('composerInput') private composerInput?: ElementRef<HTMLTextAreaElement>;

  private readonly chatBackend = inject(ChatBackendInterface);
  private readonly webllmService = inject(WebLLMService);
  private readonly ssmService = inject(SSMWebGPUService);
  private readonly modelManager = inject(ModelManagerService);
  private readonly chatHistory = inject(ChatHistoryService);
  private readonly p2pSync = inject(P2PSyncService);
  private readonly gpuCluster = inject(GPUClusterService);
  private readonly distributedInference = inject(DistributedInferenceService);
  private readonly router = inject(Router);

  readonly selectedModel = this.modelManager.selectedModel;
  readonly selectedModelId = this.modelManager.selectedModelId;
  readonly sessions = this.chatHistory.sortedSessions;
  readonly activeSession = this.chatHistory.activeSession;

  readonly isSSMModel = computed(() =>
    this.selectedModelId().includes('OxideLLM_TK_SSM'),
  );

  readonly availableGPUs = this.webllmService.availableGPUs;
  readonly selectedGPU = this.webllmService.selectedGPU;
  readonly supportsF16 = this.webllmService.supportsF16;
  readonly showGPUSelector = signal(false);

  readonly state = computed(() => {
    if (this.isSSMModel()) {
      const ssmState = this.ssmService.state();
      return {
        isInitialized: ssmState.isReady,
        isLoading: ssmState.isLoading,
        isGenerating: this._isSSMGenerating(),
        error: ssmState.error,
        loadingProgress: ssmState.isLoading
          ? { progress: ssmState.progress, text: ssmState.progressText || 'Cargando SSM…' }
          : null,
      };
    }
    return this.chatBackend.state();
  });

  readonly currentResponse = computed(() =>
    this.isSSMModel() ? this.ssmService.generatedText() : this.chatBackend.currentResponse(),
  );

  readonly messages = signal<ChatMessage[]>([]);
  readonly userInput = signal<string>('');

  readonly showSidebar = signal(
    typeof window === 'undefined' || window.matchMedia('(min-width: 768px)').matches,
  );
  readonly editingMessageIndex = signal<number | null>(null);
  readonly editingContent = signal<string>('');
  readonly showSyncModal = signal(false);
  readonly showClearConfirm = signal(false);
  readonly pendingSessionDelete = signal<string | null>(null);
  readonly copiedIndex = signal<number | null>(null);

  private readonly _isSSMGenerating = signal(false);
  private userScrolledUp = false;

  readonly isP2PConnected = this.p2pSync.isConnected;
  readonly connectedPeersCount = computed(() => this.p2pSync.connectedPeers().length);

  readonly showClusterPanel = signal(false);
  readonly isClusterActive = computed(() => this.gpuCluster.localNode() !== null);
  readonly canUseDistributed = this.distributedInference.canUseDistributed;

  private readonly p2pTaskHandler = ((event: Event) =>
    this.handleP2PTask(event as CustomEvent)) as EventListener;

  private lastLoadedSessionId = this.activeSession()?.id ?? null;

  constructor() {
    effect(() => {
      const session = this.chatHistory.activeSession();
      if (!session || session.id === this.lastLoadedSessionId) return;

      this.lastLoadedSessionId = session.id;
      this.messages.set([...session.messages]);
      this.chatBackend.resetChat();
      this.userScrolledUp = false;
    });
  }

  readonly canSend = computed(
    () =>
      this.state().isInitialized &&
      !this.state().isGenerating &&
      this.userInput().trim().length > 0,
  );

  readonly canStop = computed(() => this.state().isGenerating && !this.isSSMModel());

  readonly suggestions = [
    'Explícame cómo funciona WebGPU en este proyecto.',
    'Resume las ventajas de un modelo state-space frente a un transformer.',
    'Escribe un plan de pruebas para la inferencia distribuida P2P.',
  ];

  readonly displayMessages = computed(() => {
    const messages = [...this.messages()];
    const streaming = this.currentResponse();
    if (this.state().isGenerating && streaming) {
      messages.push({ role: 'assistant', content: streaming, timestamp: new Date() });
    }
    return messages;
  });

  ngOnInit(): void {
    const session = this.activeSession();
    if (session) {
      this.messages.set([...session.messages]);
    }

    void this.webllmService.detectAvailableGPUs();
    window.addEventListener('p2p-inference-task', this.p2pTaskHandler);
  }

  ngOnDestroy(): void {
    window.removeEventListener('p2p-inference-task', this.p2pTaskHandler);
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  onMessagesScroll(event: Event): void {
    const element = event.target as HTMLElement;
    const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
    this.userScrolledUp = distance > 96;
  }

  formatTime(value: Date): string {
    return new Intl.DateTimeFormat('es-ES', { hour: '2-digit', minute: '2-digit' }).format(
      new Date(value),
    );
  }

  p2pButtonClass(): string {
    const base =
      'inline-flex h-8 items-center gap-1.5 border px-2.5 font-mono text-[9px] uppercase tracking-[0.1em] transition-colors';
    return this.isP2PConnected()
      ? `${base} border-positive/30 bg-positive-soft text-positive`
      : `${base} border-line text-muted hover:border-ink hover:text-ink`;
  }

  clusterButtonClass(): string {
    const base =
      'inline-flex h-8 items-center gap-1.5 border px-2.5 font-mono text-[9px] uppercase tracking-[0.1em] transition-colors';
    return this.isClusterActive()
      ? `${base} border-ink bg-ink text-paper`
      : `${base} border-line text-muted hover:border-ink hover:text-ink`;
  }

  sessionTime(value: Date): string {
    return formatRelativeTime(value);
  }

  toggleSidebar(): void {
    this.showSidebar.update((open) => !open);
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
    this.showClusterPanel.update((open) => !open);
  }

  toggleGPUSelector(): void {
    this.showGPUSelector.update((open) => !open);
  }

  onGPUChange(gpuId: string): void {
    this.webllmService.selectGPU(gpuId);
    this.showGPUSelector.set(false);
  }

  async initializeModel(): Promise<void> {
    try {
      if (this.isSSMModel()) {
        await this.ssmService.loadModel();
      } else {
        await this.chatBackend.initialize();
      }

      if (!this.activeSession()) {
        this.createNewChat();
      }
    } catch (error) {
      console.error('Error initializing model:', error);
    }
  }

  async clearCacheAndRetry(): Promise<void> {
    await this.webllmService.clearModelCache();
    await this.initializeModel();
  }

  async sendMessage(): Promise<void> {
    const content = this.userInput().trim();
    if (!content || !this.canSend()) return;

    if (!this.activeSession()) {
      this.createNewChat();
    }

    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    this.messages.update((messages) => [...messages, userMessage]);
    this.userInput.set('');
    this.userScrolledUp = false;

    const sessionId = this.activeSession()?.id;
    if (sessionId) {
      this.chatHistory.updateSessionMessages(sessionId, this.messages());
    }

    try {
      let response: string;

      if (this.isSSMModel()) {
        this._isSSMGenerating.set(true);
        try {
          response = await this.ssmService.generate(content, 150, 0.8);
        } finally {
          this._isSSMGenerating.set(false);
        }
      } else {
        const history = this.messages().filter((message) => message.role !== 'system');
        const isHost = this.p2pSync.connectionMode() === 'hosting';
        const hasPeers = this.p2pSync.connectedPeers().length > 0;

        if (this.canUseDistributed()) {
          response = await this.distributedInference.runWithFallback(content);
        } else if (isHost && hasPeers) {
          response = await this.sendDistributedMessage(content, history.slice(0, -1));
        } else {
          response = await this.chatBackend.sendMessage(content, history.slice(0, -1));
        }
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      if (this.activeSession()?.id === sessionId) {
        this.messages.update((messages) => [...messages, assistantMessage]);
        if (sessionId) {
          this.chatHistory.updateSessionMessages(sessionId, this.messages());
        }
      } else if (sessionId) {
        const session = this.sessions().find((item) => item.id === sessionId);
        if (session) {
          this.chatHistory.updateSessionMessages(sessionId, [...session.messages, assistantMessage]);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      this._isSSMGenerating.set(false);
    }
  }

  stopGeneration(): void {
    this.chatBackend.abortGeneration();
  }

  useSuggestion(suggestion: string): void {
    this.userInput.set(suggestion);
    queueMicrotask(() => this.composerInput?.nativeElement.focus());
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void this.sendMessage();
    }
  }

  requestClearChat(): void {
    if (this.messages().length === 0) return;
    this.showClearConfirm.set(true);
  }

  clearChat(): void {
    const sessionId = this.activeSession()?.id;
    if (sessionId) {
      this.chatHistory.clearSessionMessages(sessionId);
    }
    this.messages.set([]);
    this.chatBackend.resetChat();
    this.showClearConfirm.set(false);
  }

  createNewChat(): void {
    const session = this.chatHistory.createSession(this.selectedModelId());
    this.lastLoadedSessionId = session.id;
    this.messages.set([]);
    this.chatBackend.resetChat();
    this.userScrolledUp = false;
  }

  loadSession(sessionId: string): void {
    this.chatHistory.setActiveSession(sessionId);
  }

  requestDeleteSession(sessionId: string, event: Event): void {
    event.stopPropagation();
    this.pendingSessionDelete.set(sessionId);
  }

  confirmDeleteSession(): void {
    const sessionId = this.pendingSessionDelete();
    if (!sessionId) return;

    this.chatHistory.deleteSession(sessionId);
    if (this.lastLoadedSessionId === sessionId) {
      const nextSession = this.activeSession();
      this.lastLoadedSessionId = nextSession?.id ?? null;
      this.messages.set(nextSession ? [...nextSession.messages] : []);
    }
    this.pendingSessionDelete.set(null);
  }

  sessionToDelete(): ChatSession | undefined {
    const sessionId = this.pendingSessionDelete();
    return this.sessions().find((session) => session.id === sessionId);
  }

  startEditMessage(index: number): void {
    const message = this.messages()[index];
    if (message.role !== 'user') return;
    this.editingMessageIndex.set(index);
    this.editingContent.set(message.content);
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

    const newMessages = this.chatHistory.editMessageAt(sessionId, index, newContent);
    this.messages.set(newMessages);
    this.cancelEdit();

    this.chatBackend.resetChat();
    try {
      const history = newMessages.filter((message) => message.role !== 'system');
      const response = await this.chatBackend.sendMessage(newContent, history.slice(0, -1));

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      if (this.activeSession()?.id === sessionId) {
        this.messages.update((messages) => [...messages, assistantMessage]);
        this.chatHistory.updateSessionMessages(sessionId, this.messages());
      } else {
        const session = this.sessions().find((item) => item.id === sessionId);
        if (session) {
          this.chatHistory.updateSessionMessages(sessionId, [...session.messages, assistantMessage]);
        }
      }
    } catch (error) {
      console.error('Error regenerating:', error);
    }
  }

  forkFromHere(index: number): void {
    const sessionId = this.activeSession()?.id;
    if (!sessionId) return;

    const newSession = this.chatHistory.forkFromMessage(sessionId, index);
    this.lastLoadedSessionId = newSession.id;
    this.messages.set([...newSession.messages]);
    this.chatBackend.resetChat();
  }

  async copyMessage(index: number, content: string): Promise<void> {
    if (!(await copyText(content))) return;
    this.copiedIndex.set(index);
    window.setTimeout(() => this.copiedIndex.set(null), 1600);
  }

  private async handleP2PTask(event: CustomEvent): Promise<void> {
    if (!this.state().isInitialized || this.isSSMModel()) return;
    const { taskId, prompt, fromId } = event.detail;

    try {
      const response = await this.chatBackend.sendMessage(prompt, []);
      this.p2pSync.sendInferenceResult(taskId, fromId, response);
    } catch (error) {
      console.error('P2P inference error:', error);
    }
  }

  private async sendDistributedMessage(
    content: string,
    history: ChatMessage[],
  ): Promise<string> {
    return new Promise((resolve) => {
      let settled = false;
      let timeoutId: number | null = null;

      const finish = (result: string): void => {
        if (settled) return;
        settled = true;
        if (timeoutId !== null) window.clearTimeout(timeoutId);
        window.removeEventListener('p2p-inference-result', resultHandler);
        resolve(result);
      };

      const taskId = this.p2pSync.sendInferenceTask(content);

      const resultHandler = (event: Event): void => {
        const detail = (event as CustomEvent).detail;
        if (detail.taskId === taskId) {
          finish(detail.result);
        }
      };

      window.addEventListener('p2p-inference-result', resultHandler);

      timeoutId = window.setTimeout(async () => {
        try {
          finish(await this.chatBackend.sendMessage(content, history));
        } catch (error) {
          finish(error instanceof Error ? error.message : 'Error de inferencia distribuida.');
        }
      }, 60_000);
    });
  }

  private scrollToBottom(): void {
    if (this.userScrolledUp) return;
    const element = this.messagesContainer?.nativeElement;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }
}
