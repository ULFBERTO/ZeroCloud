import { signal, computed, inject, Injectable, effect } from '@angular/core';
import { ThemeMode } from '../interfaces/theme.interface';

export interface AppState {
  theme: ThemeMode;
  sidebarOpen: boolean;
}

const STORAGE_KEY = 'zerocloud_theme';

@Injectable({ providedIn: 'root' })
export class AppStateService {
  private readonly _theme = signal<ThemeMode>(this.loadTheme());
  private readonly _sidebarOpen = signal<boolean>(false);

  readonly theme = this._theme.asReadonly();
  readonly sidebarOpen = this._sidebarOpen.asReadonly();

  readonly isDark = computed(() => this._theme() === 'dark');

  constructor() {
    effect(() => {
      const t = this._theme();
      localStorage.setItem(STORAGE_KEY, t);
      document.documentElement.classList.toggle('dark', t === 'dark');
    });
  }

  private loadTheme(): ThemeMode {
    if (typeof window === 'undefined') return 'dark';
    const stored = localStorage.getItem(STORAGE_KEY) as ThemeMode | null;
    return stored || 'dark';
  }

  setTheme(theme: ThemeMode): void {
    this._theme.set(theme);
  }

  toggleTheme(): void {
    this._theme.update(t => (t === 'dark' ? 'light' : 'dark'));
  }

  setSidebarOpen(open: boolean): void {
    this._sidebarOpen.set(open);
  }

  toggleSidebar(): void {
    this._sidebarOpen.update(v => !v);
  }
}
