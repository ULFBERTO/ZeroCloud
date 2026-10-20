import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

@Component({
  selector: 'app-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div [class]="classes()">
      <ng-content />
    </div>
  `,
})
export class CardComponent {
  readonly padding = input<'none' | 'sm' | 'md' | 'lg'>('md');
  readonly variant = input<'surface' | 'raised' | 'ink' | 'plain'>('surface');

  readonly classes = computed(() => {
    const padding: Record<string, string> = {
      none: 'p-0',
      sm: 'p-4',
      md: 'p-5',
      lg: 'p-7',
    };
    const variant: Record<string, string> = {
      surface: 'panel',
      raised: 'panel-raised',
      ink: 'border border-ink bg-ink text-paper',
      plain: 'border border-line bg-transparent',
    };
    return ['rounded-[2px]', padding[this.padding()], variant[this.variant()]].join(' ');
  });
}
