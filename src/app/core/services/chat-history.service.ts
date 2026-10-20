import { Injectable, computed, signal } from '@angular/core';
import { ChatMessage } from '../interfaces/chat-backend.interface';
import { readJson, readStorage, removeStorage, writeJson, writeStorage } from '../utils/storage';
import { truncate } from '../utils/format';

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
const NEW_CHAT_TITLE = 'Nueva conversación';

@Injectable({ providedIn: 'root' })
export class ChatHistoryService {
  private readonly _sessions = signal<ChatSession[]>([]);
  private readonly _activeSessionId = signal<string | null>(null);

  readonly sessions = this._sessions.asReadonly();
  readonly activeSessionId = this._activeSessionId.asReadonly();

  readonly activeSession = computed(() =>
    this._sessions().find((session) => session.id === this._activeSessionId()),
  );

  readonly sortedSessions = computed(() =>
    [...this._sessions()].sort(
      (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
    ),
  );

  readonly totalMessages = computed(() =>
    this._sessions().reduce((total, session) => total + session.messages.length, 0),
  );

  constructor() {
    this.loadFromStorage();
  }

  createSession(modelId: string): ChatSession {
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title: NEW_CHAT_TITLE,
      modelId,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this._sessions.update((sessions) => [session, ...sessions]);
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
      sessions.map((session) => {
        if (session.id !== sessionId) return session;

        let title = session.title;
        if (title === NEW_CHAT_TITLE) {
          const firstUserMessage = messages.find((message) => message.role === 'user');
          if (firstUserMessage) {
            title = truncate(firstUserMessage.content, 48);
          }
        }

        return { ...session, messages, title, updatedAt: new Date() };
      }),
    );
    this.saveToStorage();
  }

  clearSessionMessages(sessionId: string): void {
    this.updateSessionMessages(sessionId, []);
  }

  forkFromMessage(sessionId: string, messageIndex: number): ChatSession {
    const session = this._sessions().find((item) => item.id === sessionId);
    if (!session) throw new Error('Session not found');

    const newSession: ChatSession = {
      id: crypto.randomUUID(),
      title: `${session.title} (fork)`,
      modelId: session.modelId,
      messages: session.messages.slice(0, messageIndex + 1),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this._sessions.update((sessions) => [newSession, ...sessions]);
    this._activeSessionId.set(newSession.id);
    this.saveToStorage();
    return newSession;
  }

  editMessageAt(sessionId: string, messageIndex: number, newContent: string): ChatMessage[] {
    const session = this._sessions().find((item) => item.id === sessionId);
    if (!session) throw new Error('Session not found');

    const messages = session.messages.slice(0, messageIndex);
    messages.push({
      ...session.messages[messageIndex],
      content: newContent,
      timestamp: new Date(),
    });

    this._sessions.update((sessions) =>
      sessions.map((item) =>
        item.id === sessionId ? { ...item, messages, updatedAt: new Date() } : item,
      ),
    );
    this.saveToStorage();
    return messages;
  }

  renameSession(sessionId: string, newTitle: string): void {
    this._sessions.update((sessions) =>
      sessions.map((session) =>
        session.id === sessionId
          ? { ...session, title: newTitle.trim() || NEW_CHAT_TITLE, updatedAt: new Date() }
          : session,
      ),
    );
    this.saveToStorage();
  }

  deleteSession(sessionId: string): void {
    this._sessions.update((sessions) => sessions.filter((session) => session.id !== sessionId));

    if (this._activeSessionId() === sessionId) {
      const remaining = this.sortedSessions();
      this._activeSessionId.set(remaining[0]?.id ?? null);
    }
    this.saveToStorage();
  }

  clearAllSessions(): void {
    this._sessions.set([]);
    this._activeSessionId.set(null);
    this.saveToStorage();
  }

  exportSession(sessionId: string): string {
    const session = this._sessions().find((item) => item.id === sessionId);
    if (!session) throw new Error('Session not found');
    return JSON.stringify(session, null, 2);
  }

  importSession(jsonData: string): ChatSession {
    const data = JSON.parse(jsonData);
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title: data.title || 'Chat importado',
      modelId: data.modelId || 'unknown',
      messages: (data.messages || []).map((message: ChatMessage) => ({
        ...message,
        timestamp: new Date(message.timestamp),
      })),
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this._sessions.update((sessions) => [session, ...sessions]);
    this.saveToStorage();
    return session;
  }

  private loadFromStorage(): void {
    const stored = readJson<ChatSession[]>(STORAGE_KEY, []);

    const sessions = stored
      .filter((session) => session?.id && Array.isArray(session.messages))
      .map((session) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        updatedAt: new Date(session.updatedAt),
        messages: session.messages.map((message) => ({
          ...message,
          timestamp: new Date(message.timestamp),
        })),
      }));

    this._sessions.set(sessions);

    const activeId = readStorage(ACTIVE_SESSION_KEY);
    if (activeId && sessions.some((session) => session.id === activeId)) {
      this._activeSessionId.set(activeId);
    }
  }

  private saveToStorage(): void {
    writeJson(STORAGE_KEY, this._sessions());
    const activeId = this._activeSessionId();
    if (activeId) {
      writeStorage(ACTIVE_SESSION_KEY, activeId);
    } else {
      removeStorage(ACTIVE_SESSION_KEY);
    }
  }
}
