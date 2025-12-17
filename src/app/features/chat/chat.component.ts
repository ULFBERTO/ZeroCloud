import {
  Component,
  inject,
  signal,
  computed,
  ViewChild,
  ElementRef,
  AfterViewChecked,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  ChatBackendInterface,
  ChatMessage,
} from '../../core/interfaces/chat-backend.interface';
import { WebLLMService } from '../../core/services/webllm.service';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule],
  providers: [
    { provide: ChatBackendInterface, useClass: WebLLMService },
  ],
  templateUrl: './chat.component.html',
})
export class ChatComponent implements AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!: ElementRef;

  private readonly chatBackend = inject(ChatBackendInterface);

  readonly state = this.chatBackend.state;
  readonly currentResponse = this.chatBackend.currentResponse;
  readonly messages = signal<ChatMessage[]>([]);
  readonly userInput = signal<string>('');

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


  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  async initializeModel(): Promise<void> {
    try {
      await this.chatBackend.initialize();
    } catch (error) {
      console.error('Error initializing model:', error);
    }
  }

  async sendMessage(): Promise<void> {
    const content = this.userInput().trim();
    if (!content || !this.canSend()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    this.messages.update((msgs) => [...msgs, userMessage]);
    this.userInput.set('');

    try {
      const history = this.messages().filter((m) => m.role !== 'system');
      const response = await this.chatBackend.sendMessage(content, history.slice(0, -1));

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      this.messages.update((msgs) => [...msgs, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  }

  stopGeneration(): void {
    this.chatBackend.abortGeneration();
  }

  clearChat(): void {
    this.messages.set([]);
    this.chatBackend.resetChat();
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  private scrollToBottom(): void {
    try {
      const el = this.messagesContainer?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    } catch {}
  }
}
