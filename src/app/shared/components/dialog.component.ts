import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { IconComponent } from './icon.component';

@Component({
  selector: 'app-dialog',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [IconComponent],
  template: `
    @if (open()) {
      <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
        <button
          type="button"
          class="animate-fade absolute inset-0 cursor-default bg-ink/35 backdrop-blur-[2px]"
          aria-label="Cerrar diálogo"
          (click)="close.emit()"
        ></button>
        <div
          class="animate-rise relative w-full max-w-lg border border-line bg-surface p-6 shadow-float"
          role="dialog"
          aria-modal="true"
          [attr.aria-label]="title()"
        >
          <div class="mb-4 flex items-start justify-between gap-4">
            <h3 class="text-lg font-semibold tracking-[-0.02em] text-ink">{{ title() }}</h3>
            <button
              type="button"
              class="rounded-[2px] p-1.5 text-muted transition-colors hover:bg-ink/5 hover:text-ink"
              aria-label="Cerrar"
              (click)="close.emit()"
            >
              <app-icon name="close" [size]="16" />
            </button>
          </div>
          <div class="text-sm leading-relaxed text-muted">
            <ng-content />
          </div>
          <div class="mt-6 flex justify-end gap-2">
            <ng-content select="[footer]" />
          </div>
        </div>
      </div>
    }
  `,
})
export class DialogComponent {
  readonly open = input<boolean>(false);
  readonly title = input<string>('');
  readonly close = output<void>();
}
