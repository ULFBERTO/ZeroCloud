import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { filter } from 'rxjs';
import { PRIMARY_NAV, RESOURCE_NAV } from '../core/config/navigation';
import { ChatHistoryService } from '../core/services/chat-history.service';
import { ModelManagerService } from '../core/services/model-manager.service';
import { AppStateService } from '../core/state';
import { formatBytes, percentage } from '../core/utils/format';
import { AvatarComponent, IconComponent } from '../shared/components';

@Component({
  selector: 'app-shell',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    RouterOutlet,
    RouterLink,
    RouterLinkActive,
    IconComponent,
    AvatarComponent,
  ],
  templateUrl: './app-shell.component.html',
})
export class AppShellComponent {
  private readonly router = inject(Router);
  readonly appState = inject(AppStateService);
  private readonly modelManager = inject(ModelManagerService);
  private readonly chatHistory = inject(ChatHistoryService);

  readonly primaryNav = PRIMARY_NAV;
  readonly resourceNav = RESOURCE_NAV;
  readonly mobileNavOpen = this.appState.mobileNavOpen;
  readonly selectedModel = this.modelManager.selectedModel;

  readonly storage = signal<{ used: string; quota: string; percent: number } | null>(null);

  readonly storageLabel = computed(() => {
    const current = this.storage();
    if (!current) return 'Calculando…';
    return `${current.used} de ${current.quota}`;
  });

  constructor() {
    this.router.events
      .pipe(
        filter((event) => event instanceof NavigationEnd),
        takeUntilDestroyed(),
      )
      .subscribe(() => {
        this.appState.closeMobileNav();
        void this.loadStorage();
      });

    void this.loadStorage();
  }

  closeMobileNav(): void {
    this.appState.closeMobileNav();
  }

  newChat(): void {
    this.chatHistory.createSession(this.modelManager.selectedModelId());
    void this.router.navigate(['/playground']);
  }

  private async loadStorage(): Promise<void> {
    const usage = await this.modelManager.getStorageUsage();
    if (!usage) return;
    this.storage.set({
      used: formatBytes(usage.used),
      quota: formatBytes(usage.quota),
      percent: percentage(usage.used, usage.quota),
    });
  }
}
