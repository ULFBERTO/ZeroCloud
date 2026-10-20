import { ChangeDetectionStrategy, Component, booleanAttribute, computed, input } from '@angular/core';

export type BadgeVariant = 'neutral' | 'success' | 'warning' | 'error' | 'info' | 'ink' | 'outline';

@Component({
  selector: 'app-badge',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span [class]="classes()">
      @if (dot()) {
        <span [class]="dotClass()"></span>
      }
      {{ label() }}
    </span>
  `,
})
export class BadgeComponent {
  readonly label = input<string>('');
  readonly variant = input<BadgeVariant>('neutral');
  readonly dot = input(false, { transform: booleanAttribute });

  readonly classes = computed(() => {
    const base =
      'inline-flex items-center gap-1.5 rounded-[2px] border px-2 py-[3px] font-mono text-[9px] font-medium uppercase tracking-[0.12em] leading-none';
    const variants: Record<BadgeVariant, string> = {
      neutral: 'border-line bg-surface text-muted',
      success: 'border-positive/25 bg-positive-soft text-positive',
      warning: 'border-warning/25 bg-warning-soft text-warning',
      error: 'border-negative/25 bg-negative-soft text-negative',
      info: 'border-accent/20 bg-accent-soft text-accent',
      ink: 'border-ink bg-ink text-paper',
      outline: 'border-line-strong bg-transparent text-graphite',
    };
    return [base, variants[this.variant()]].join(' ');
  });

  readonly dotClass = computed(() => {
    const colors: Record<BadgeVariant, string> = {
      neutral: 'bg-faint',
      success: 'bg-positive',
      warning: 'bg-warning',
      error: 'bg-negative',
      info: 'bg-accent',
      ink: 'bg-paper',
      outline: 'bg-graphite',
    };
    return `h-1 w-1 rounded-full ${colors[this.variant()]}`;
  });
}
