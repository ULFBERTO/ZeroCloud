import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { IconComponent, IconName } from './icon.component';

export type ButtonVariant = 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'link';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'icon';

@Component({
  selector: 'app-button',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [IconComponent],
  template: `
    <button
      [type]="type()"
      [disabled]="disabled() || loading()"
      [attr.aria-label]="ariaLabel() || null"
      [attr.aria-busy]="loading() ? 'true' : null"
      [class]="classes()"
      (click)="handleClick()"
    >
      @if (loading()) {
        <span class="h-3 w-3 animate-spin rounded-full border border-current border-t-transparent"></span>
      } @else if (icon() && iconPosition() === 'left') {
        <app-icon [name]="icon()!" [size]="iconSize()" />
      }

      @if (label()) {
        <span class="truncate">{{ label() }}</span>
      }

      @if (!loading() && icon() && iconPosition() === 'right') {
        <app-icon [name]="icon()!" [size]="iconSize()" />
      }
    </button>
  `,
})
export class ButtonComponent {
  readonly label = input<string>('');
  readonly type = input<'button' | 'submit' | 'reset'>('button');
  readonly disabled = input<boolean>(false);
  readonly loading = input<boolean>(false);
  readonly fullWidth = input<boolean>(false);
  readonly variant = input<ButtonVariant>('primary');
  readonly size = input<ButtonSize>('md');
  readonly icon = input<IconName | null>(null);
  readonly iconPosition = input<'left' | 'right'>('right');
  readonly ariaLabel = input<string>('');

  readonly clicked = output<void>();

  readonly iconSize = computed(() => (this.size() === 'sm' ? 14 : 16));

  readonly classes = computed(() => {
    const base =
      'inline-flex select-none items-center justify-center gap-2 rounded-[3px] font-medium leading-none transition-colors duration-150 disabled:pointer-events-none disabled:opacity-40';
    const sizes: Record<ButtonSize, string> = {
      sm: 'h-8 px-3 text-[11px]',
      md: 'h-10 px-4 text-[13px]',
      lg: 'h-12 px-5 text-sm',
      icon: 'h-9 w-9 p-0',
    };
    const variants: Record<ButtonVariant, string> = {
      primary: 'bg-ink text-paper hover:bg-graphite',
      secondary: 'border border-line bg-surface text-ink hover:border-line-strong hover:bg-[#efefe9]',
      outline: 'border border-ink bg-transparent text-ink hover:bg-ink hover:text-paper',
      ghost: 'text-muted hover:bg-ink/5 hover:text-ink',
      danger: 'border border-negative/25 bg-negative-soft text-negative hover:bg-negative hover:text-white',
      link: 'h-auto px-0 text-accent hover:underline underline-offset-4',
    };
    return [base, sizes[this.size()], variants[this.variant()], this.fullWidth() ? 'w-full' : ''].join(
      ' ',
    );
  });

  handleClick(): void {
    if (this.disabled() || this.loading()) return;
    this.clicked.emit();
  }
}
