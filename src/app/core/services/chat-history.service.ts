import { Injectable, signal, computed } from '@angular/core';
import { ChatMessage } from '../interfaces/chat-backend.interface';

export interface ChatSession {
  id: string;
  title: string;
  modelId: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

const STORAGE_KEY = 'webllm_chat_sessions';
const ACTIVE_SESSION_KEY = 'webllm_active_session';

@Injectable({ providedIn: 'root' })
export class ChatHistoryService {
  private readonly _sessions = signal<ChatSession[]>([]);
  private readonly _activeSessionId = signal<string | null>(null);

  readonly sessions = this._sessions.asReadonly();
  readonly activeSessionId = this._activeSessionId.asReadonly();

  readonly activeSession = computed(() =>
    this._sessions().find((s) => s.id === this._activeSessionId())
  );

  readonly sortedSessions = computed(() =>
    [...this._sessions()].sort(
      (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    )
  );

  constructor() {
    this.loadFromStorage();
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const sessions = JSON.parse(stored).map((s: ChatSession) => ({
          ...s,
          createdAt: new Date(s.createdAt),
          updatedAt: new Date(s.updatedAt),
          messages: s.messages.map((m) => ({ ...m, timestamp: new Date(m.timestamp) })),
        }));
        this._sessions.set(sessions);
      }

      const activeId = localStorage.getItem(ACTIVE_SESSION_KEY);
      if (activeId && this._sessions().some((s) => s.id === activeId)) {
        this._activeSessionId.set(activeId);
      }
    } catch (e) {
      console.error('Error loading chat history:', e);
    }
  }

  private saveToStorage(): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this._sessions()));
    const activeId = this._activeSessionId();
    if (activeId) {
      localStorage.setItem(ACTIVE_SESSION_KEY, activeId);
    } else {
      localStorage.removeItem(ACTIVE_SESSION_KEY);
    }
  }


  createSession(modelId: string): ChatSession {
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title: 'Nueva conversación',
      modelId,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this._sessions.update((s) => [session, ...s]);
    this._activeSessionId.set(session.id);
    this.saveToStorage();
    return session;
  }

  setActiveSession(sessionId: string | null): void {
    this._activeSessionId.set(sessionId);
    this.saveToStorage();
  }

  updateSessionMessages(sessionId: string, messages: ChatMessage[]): void {
    this._sessions.update((sessions) =>
      sessions.map((s) => {
        if (s.id !== sessionId) return s;

        // Auto-generar título del primer mensaje del usuario
        let title = s.title;
        if (title === 'Nueva conversación' && messages.length > 0) {
          const firstUserMsg = messages.find((m) => m.role === 'user');
          if (firstUserMsg) {
            title = firstUserMsg.content.slice(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '');
          }
        }

        return { ...s, messages, title, updatedAt: new Date() };
      })
    );
    this.saveToStorage();
  }

  // Volver a un punto específico del chat (fork)
  forkFromMessage(sessionId: string, messageIndex: number): ChatSession {
    const session = this._sessions().find((s) => s.id === sessionId);
    if (!session) throw new Error('Session not found');

    const newSession: ChatSession = {
      id: crypto.randomUUID(),
      title: `${session.title} (fork)`,
      modelId: session.modelId,
      messages: session.messages.slice(0, messageIndex + 1),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this._sessions.update((s) => [newSession, ...s]);
    this._activeSessionId.set(newSession.id);
    this.saveToStorage();
    return newSession;
  }

  // Editar un mensaje y regenerar desde ahí
  editMessageAt(sessionId: string, messageIndex: number, newContent: string): ChatMessage[] {
    const session = this._sessions().find((s) => s.id === sessionId);
    if (!session) throw new Error('Session not found');

    // Cortar mensajes hasta el editado y actualizar contenido
    const newMessages = session.messages.slice(0, messageIndex);
    const editedMessage: ChatMessage = {
      ...session.messages[messageIndex],
      content: newContent,
      timestamp: new Date(),
    };
    newMessages.push(editedMessage);

    this._sessions.update((sessions) =>
      sessions.map((s) =>
        s.id === sessionId ? { ...s, messages: newMessages, updatedAt: new Date() } : s
      )
    );
    this.saveToStorage();
    return newMessages;
  }


  renameSession(sessionId: string, newTitle: string): void {
    this._sessions.update((sessions) =>
      sessions.map((s) =>
        s.id === sessionId ? { ...s, title: newTitle, updatedAt: new Date() } : s
      )
    );
    this.saveToStorage();
  }

  deleteSession(sessionId: string): void {
    this._sessions.update((s) => s.filter((session) => session.id !== sessionId));
    if (this._activeSessionId() === sessionId) {
      const remaining = this._sessions();
      this._activeSessionId.set(remaining.length > 0 ? remaining[0].id : null);
    }
    this.saveToStorage();
  }

  clearAllSessions(): void {
    this._sessions.set([]);
    this._activeSessionId.set(null);
    this.saveToStorage();
  }

  exportSession(sessionId: string): string {
    const session = this._sessions().find((s) => s.id === sessionId);
    if (!session) throw new Error('Session not found');
    return JSON.stringify(session, null, 2);
  }

  importSession(jsonData: string): ChatSession {
    const data = JSON.parse(jsonData);
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title: data.title || 'Imported chat',
      modelId: data.modelId || 'unknown',
      messages: (data.messages || []).map((m: ChatMessage) => ({
        ...m,
        timestamp: new Date(m.timestamp),
      })),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this._sessions.update((s) => [session, ...s]);
    this.saveToStorage();
    return session;
  }
}
